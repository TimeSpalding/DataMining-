"""
recommender.py — LAYER 4: Inference API
=========================================
Class LocalRecommender — tích hợp:
  ✦ LightGCN embeddings (user_vectors.npy + item_vectors.npy) — mmap
  ✦ TF-IDF Content Index nâng cao (thêm dominant_genre vào text)
  ✦ SQLite Knowledge Base (churn_risk, persona, dominant_genre)
  ✦ Genre-Aware Scoring: +20% điểm nếu genre khớp với sở thích user
  ✦ Churn Protection: user sắp rời (>70%) → chặn exploration, đẩy quen thuộc

Sử dụng:
    from recommender import LocalRecommender
    rec = LocalRecommender()
    rec.recommend_hybrid("user_id_123", n=10)
"""
import os, math, pickle, sqlite3, warnings
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, KB_PATH, RECOMMENDER_CONFIG

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize một vector hoặc matrix."""
    vec = np.atleast_2d(vec).astype(np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.where(norms > 0, norms, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL RECOMMENDER
# ──────────────────────────────────────────────────────────────────────────────

class LocalRecommender:
    """
    Hybrid recommender tích hợp 4 tầng:
    - LightGCN collaborative embedding
    - TF-IDF content similarity (genre-enriched)
    - Churn-aware reranking
    - Genre-boost scoring
    """

    # ── INIT ──────────────────────────────────────────────────────────────────

    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        kb_path:   str = KB_PATH,
        cold_threshold: int = RECOMMENDER_CONFIG["cold_threshold"],
    ):
        self.model_dir      = model_dir
        self.kb_path        = kb_path
        self.cold_threshold = cold_threshold
        self.cfg            = RECOMMENDER_CONFIG
        self._db_conn: Optional[sqlite3.Connection] = None

        print("=" * 55)
        print(" LocalRecommender — Loading...")
        print("=" * 55)
        self._load_embeddings()
        self._load_mappings()
        self._load_matrices()
        self._build_content_index()
        self._build_artist_index()
        self._compute_trending()
        self._user_index = None
        print("\n✅ Recommender sẵn sàng!")
        print(f"   Users: {len(self.idx2user):,} | Items: {len(self.idx2item):,}")
        print(f"   KB: {kb_path}")

    # ── DATABASE ──────────────────────────────────────────────────────────────

    @property
    def db(self) -> sqlite3.Connection:
        if self._db_conn is None:
            if not os.path.exists(self.kb_path):
                raise FileNotFoundError(
                    f"mappings.db không tồn tại: {self.kb_path}\n"
                    "Chạy 05_build_knowledge_base.py trước!"
                )
            self._db_conn = sqlite3.connect(self.kb_path, check_same_thread=False)
            self._db_conn.row_factory = sqlite3.Row
        return self._db_conn

    def _query_user(self, user_id: str) -> dict:
        """Lấy churn_risk, dominant_genre, persona_label của một user."""
        row = self.db.execute(
            "SELECT churn_risk, dominant_genre, persona_label, churn_tier "
            "FROM users WHERE user_id = ? LIMIT 1",
            (user_id,)
        ).fetchone()
        if row is None:
            return {"churn_risk": 0.0, "dominant_genre": None,
                    "persona_label": "Unknown", "churn_tier": "LOW"}
        return dict(row)

    def _query_item_genre(self, msid: str) -> Optional[str]:
        """Lấy dominant_genre của một track."""
        row = self.db.execute(
            "SELECT dominant_genre FROM items WHERE msid = ? LIMIT 1", (msid,)
        ).fetchone()
        return row["dominant_genre"] if row else None

    def _query_artist_genre(self, artist_name: str) -> Optional[str]:
        """Lấy dominant_genre của một nghệ sĩ."""
        row = self.db.execute(
            "SELECT dominant_genre FROM artists WHERE artist_name_lower = ? LIMIT 1",
            (artist_name.lower().strip(),)
        ).fetchone()
        return row["dominant_genre"] if row else None

    # ── LOAD MODELS ───────────────────────────────────────────────────────────

    def _load_embeddings(self):
        print("  [1/6] Loading embeddings (mmap)...")
        self.user_vectors = np.load(
            os.path.join(self.model_dir, "user_vectors.npy"), mmap_mode="r"
        )
        self.item_vectors = np.load(
            os.path.join(self.model_dir, "item_vectors.npy"), mmap_mode="r"
        )
        self.dim = self.user_vectors.shape[1]
        # FAISS index chính cho items
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.array(self.item_vectors, dtype=np.float32))
        print(f"     dim={self.dim} | users={self.user_vectors.shape[0]:,} | items={self.item_vectors.shape[0]:,}")

    def _load_mappings(self):
        print("  [2/6] Loading index mappings...")
        with open(os.path.join(self.model_dir, "index_mappings.pkl"), "rb") as f:
            m = pickle.load(f)
        self.user2idx          = m["user2idx"]
        self.item2idx          = m["item2idx"]
        self.idx2user          = m["idx2user"]
        self.idx2item          = m["idx2item"]
        self.item_meta         = m["item_meta"]
        self.user_item_ts_matrix = defaultdict(dict, m.get("user_item_ts_matrix", {}))
        self.global_max_ts     = m.get("global_max_ts", 0.0)

    def _load_matrices(self):
        print("  [3/6] Loading sparse matrices...")
        self.train_matrix = sp.load_npz(os.path.join(self.model_dir, "train_matrix.npz"))
        try:
            self.test_matrix = sp.load_npz(os.path.join(self.model_dir, "test_matrix.npz"))
        except Exception:
            self.test_matrix = None

    # ── CONTENT INDEX (TF-IDF Genre-Enriched) ────────────────────────────────

    def _build_content_index(self):
        """
        Nâng cấp TF-IDF: nhồi dominant_genre của nghệ sĩ vào văn bản.
        Text = "{artist} {artist} {track} {dominant_genre} {dominant_genre}"
        → Các bài cùng thể loại sẽ gần nhau hơn trong không gian vector.
        """
        print("  [4/6] Building genre-enriched TF-IDF content index...")

        # Tải artist→genre mapping trước từ DB (batch query để tránh N+1)
        artist_genre_map: dict = {}
        try:
            rows = self.db.execute("SELECT artist_name_lower, dominant_genre FROM artists").fetchall()
            artist_genre_map = {row["artist_name_lower"]: row["dominant_genre"] for row in rows}
        except Exception:
            pass  # DB chưa có → không có genre info, vẫn chạy được

        corpus, msid_list = [], []
        for msid, meta in self.item_meta.items():
            artist = meta.get("artist_name", "").strip()
            track  = meta.get("track_name",  "").strip()
            genre  = artist_genre_map.get(artist.lower(), "") or ""
            # Nhân đôi artist & genre để tăng trọng số
            text   = f"{artist} {artist} {track} {genre} {genre}".strip()
            corpus.append(text)
            msid_list.append(msid)

        self._tfidf     = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2))
        tfidf_matrix    = self._tfidf.fit_transform(corpus)
        # L2-normalize để cosine similarity = inner product
        from sklearn.preprocessing import normalize as sk_normalize
        tfidf_dense     = sk_normalize(tfidf_matrix, norm="l2").toarray().astype(np.float32)

        self._cold_item_msids = msid_list
        self._cold_item_vecs  = tfidf_dense

        # FAISS index dùng cho cold-start search
        self._cold_item_index = faiss.IndexFlatIP(tfidf_dense.shape[1])
        self._cold_item_index.add(tfidf_dense)

        # Lookup pos trong cold index theo msid
        self._msid_to_cold_pos = {msid: i for i, msid in enumerate(msid_list)}

        # Warm items (trong LightGCN model)
        warm_iids, warm_vecs = [], []
        for i, msid in enumerate(msid_list):
            if msid in self.item2idx:
                warm_iids.append(self.item2idx[msid])
                warm_vecs.append(tfidf_dense[i])
        self._content_iids = np.array(warm_iids, dtype=np.int32)
        if warm_vecs:
            warm_arr = np.stack(warm_vecs).astype(np.float32)
            self._content_index = faiss.IndexFlatIP(warm_arr.shape[1])
            self._content_index.add(warm_arr)
        else:
            self._content_index = None
        print(f"     corpus={len(corpus):,} | cold_index_dim={tfidf_dense.shape[1]:,}")

    # ── ARTIST INDEX ──────────────────────────────────────────────────────────

    def _build_artist_index(self):
        print("  [5/6] Building artist index...")
        self._artist2items = defaultdict(list)
        self._iid_to_artist = {}
        for msid, meta in self.item_meta.items():
            if msid in self.item2idx:
                iid     = self.item2idx[msid]
                a_lower = meta.get("artist_name", "").lower().strip()
                self._artist2items[a_lower].append(iid)
                self._iid_to_artist[iid] = a_lower

        artist_names = list(self._artist2items.keys())
        self._artist_names = artist_names
        artist_vecs  = []
        for name in artist_names:
            iids  = self._artist2items[name][:30]
            avg   = self.item_vectors[iids].mean(axis=0)
            artist_vecs.append(avg / (np.linalg.norm(avg) + 1e-8))

        if artist_vecs:
            self._artist_vecs = np.stack(artist_vecs).astype(np.float32)
            self._artist_index= faiss.IndexFlatIP(self._artist_vecs.shape[1])
            self._artist_index.add(self._artist_vecs)
            self._artist_name_to_pos = {n: i for i, n in enumerate(artist_names)}
        else:
            self._artist_vecs  = np.array([])
            self._artist_index = None
            self._artist_name_to_pos = {}

    # ── TRENDING ──────────────────────────────────────────────────────────────

    def _compute_trending(self):
        print("  [6/6] Computing trending scores...")
        if self.global_max_ts == 0:
            # Fallback: popularity-based
            self._trending_scores = np.asarray(
                self.train_matrix.sum(axis=0)
            ).flatten().astype(np.float32)
        else:
            hl = self.cfg["trending_decay_days"] * 86400
            scores = np.zeros(len(self.idx2item), dtype=np.float32)
            for uid in range(self.train_matrix.shape[0]):
                ts_dict = self.user_item_ts_matrix.get(uid, {})
                for iid_sparse, v in zip(
                    self.train_matrix[uid].indices,
                    self.train_matrix[uid].data,
                ):
                    last_ts = ts_dict.get(int(iid_sparse), 0)
                    if last_ts > 0:
                        decay = math.exp(-(self.global_max_ts - last_ts) / hl)
                        scores[int(iid_sparse)] += float(v) * decay
                    else:
                        scores[int(iid_sparse)] += float(v)
            self._trending_scores = scores

    # ── UTILS ─────────────────────────────────────────────────────────────────

    def _listened_set(self, user_id: Optional[str]) -> set:
        if user_id is None:
            return set()
        uid = self.user2idx.get(str(user_id))
        if uid is None:
            return set()
        return set(self.train_matrix[uid].indices.tolist())

    def _get_user_tier(self, user_id: str):
        """Trả về (tier_label, uid_int) — tier dựa trên số lượt nghe."""
        uid = self.user2idx.get(str(user_id))
        if uid is None:
            return "cold", None
        n_plays = int(self.train_matrix[uid].nnz)
        if n_plays >= 50:
            return "warm", uid
        return "cold_ish", uid

    def _to_row(self, msid: str, score: float, in_model: bool = True) -> dict:
        meta = self.item_meta.get(msid, {})
        return {
            "msid":        msid,
            "track_name":  meta.get("track_name", "Unknown"),
            "artist_name": meta.get("artist_name", "Unknown"),
            "score":       round(score, 4),
            "in_model":    in_model,
        }

    def _mmr_rerank(
        self,
        cand_ids: List[int],
        cand_scores: List[float],
        n: int,
        lambda_: float = 0.5,
    ):
        """
        Maximal Marginal Relevance — cân bằng relevance và diversity.
        lambda_ = 1.0 → thuần relevance; lambda_ = 0.0 → thuần diversity.
        """
        if not cand_ids:
            return [], []
        cands = list(zip(cand_ids, cand_scores))
        selected, selected_vecs = [], []
        norm_scores = np.array(cand_scores, dtype=np.float32)
        if norm_scores.max() > 0:
            norm_scores = norm_scores / norm_scores.max()

        while len(selected) < n and cands:
            if not selected_vecs:
                best_idx = int(np.argmax([s for _, s in cands]))
            else:
                mmr_scores = []
                for i, (iid, s) in enumerate(cands):
                    v     = self.item_vectors[iid]
                    sim_s = max(float(np.dot(v, sv)) for sv in selected_vecs)
                    mmr_scores.append(lambda_ * norm_scores[cand_ids.index(iid)] - (1 - lambda_) * sim_s)
                best_idx = int(np.argmax(mmr_scores))
            iid, sc    = cands.pop(best_idx)
            selected.append((iid, sc))
            selected_vecs.append(_normalize(self.item_vectors[iid])[0])

        ids    = [i for i, _ in selected]
        scores = [s for _, s in selected]
        return ids, scores

    def _text_to_content_vec(self, text: str) -> np.ndarray:
        vec = self._tfidf.transform([text]).toarray().astype(np.float32)
        return _normalize(vec)

    def _proxy_vector_from_items(self, iids: np.ndarray, weights: Optional[np.ndarray] = None):
        vecs = self.item_vectors[iids].astype(np.float32)
        if weights is not None:
            avg = (vecs * weights[:, None]).sum(axis=0)
        else:
            avg = vecs.mean(axis=0)
        return _normalize(avg)

    def _get_user_content_vec(self, user_id: str) -> Optional[np.ndarray]:
        uid = self.user2idx.get(str(user_id))
        if uid is None:
            return None
        iids = self.train_matrix[uid].indices
        if len(iids) == 0:
            return None
        cold_positions = [self._msid_to_cold_pos[self.idx2item[i]]
                          for i in iids if self.idx2item[i] in self._msid_to_cold_pos]
        if not cold_positions:
            return None
        avg = self._cold_item_vecs[cold_positions].mean(axis=0)
        return _normalize(avg)

    # ──────────────────────────────────────────────────────────────────────────
    # CORE: GENRE-BOOSTED + CHURN-AWARE SCORING
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_genre_boost(
        self,
        cand_ids: List[int],
        cand_scores: List[float],
        user_genre: Optional[str],
        boost: float,
    ) -> List[float]:
        """
        Nhân điểm LightGCN lên `boost` lần nếu genre bài hát khớp genre user.
        Cải tiến: batch query thay vì N+1 queries.
        """
        if not user_genre or not cand_ids:
            return cand_scores

        # Batch query items
        msids = [self.idx2item[i] for i in cand_ids]
        placeholders = ",".join("?" * len(msids))
        rows = self.db.execute(
            f"SELECT msid, dominant_genre FROM items WHERE msid IN ({placeholders})",
            msids,
        ).fetchall()
        genre_map = {row["msid"]: row["dominant_genre"] for row in rows}

        boosted = []
        for iid, sc in zip(cand_ids, cand_scores):
            msid         = self.idx2item[iid]
            item_genre   = genre_map.get(msid) or self._query_item_genre(msid)
            if item_genre and item_genre == user_genre:
                boosted.append(sc * boost)
            else:
                boosted.append(sc)
        return boosted

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def recommend_hybrid(
        self,
        user_id_str: str,
        n: int = 10,
        content_alpha: Optional[float] = None,
        artist_limit: int = 3,
    ) -> pd.DataFrame:
        """
        Gợi ý hybrid = LightGCN + TF-IDF Content.
        
        Hai lớp bảo vệ tự động:
          1. Genre Boost  : +20% điểm nếu bài hát khớp thể loại ưa thích của user
          2. Churn Guard  : Nếu churn_risk > 70% → tắt exploration, boost quen thuộc
        """
        print(f"\n[recommend_hybrid] user={user_id_str}")

        # Lấy thông tin user từ Knowledge Base
        user_info = self._query_user(user_id_str)
        churn_risk    = float(user_info.get("churn_risk", 0) or 0)
        user_genre    = user_info.get("dominant_genre")
        persona       = user_info.get("persona_label", "Unknown")
        churn_tier    = user_info.get("churn_tier", "LOW")

        print(f"  persona={persona} | genre={user_genre} | churn={churn_risk:.1f}% [{churn_tier}]")

        # ── Churn Guard ───────────────────────────────────────────────────────
        if content_alpha is None:
            if churn_risk >= self.cfg["churn_risk_threshold"]:
                content_alpha = 0.0   # Tắt exploration
                print(f"  ⚠ CHURN GUARD: content_alpha→0 (user sắp rời!)")
            else:
                content_alpha = self.cfg["default_content_alpha"]

        # ── LightGCN Scoring ──────────────────────────────────────────────────
        tier, uid = self._get_user_tier(user_id_str)
        if uid is None:
            # Cold-start: dùng thuần content hoặc popular
            print(f"  Cold-start user → content fallback")
            return self.recommend_cold_content(user_id_str, n=n)

        fetch_k = n * 6
        lgcn_scores_raw, lgcn_indices = self.index.search(
            np.array(self.user_vectors[uid:uid+1], dtype=np.float32), fetch_k
        )
        listened = self._listened_set(user_id_str)
        train_seen = set(self.train_matrix[uid].indices.tolist())

        valid_mask   = ~np.isin(lgcn_indices[0], list(train_seen))
        valid_cands  = lgcn_indices[0][valid_mask].tolist()
        valid_scores = lgcn_scores_raw[0][valid_mask].tolist()
        valid_cands  = [i for i in valid_cands if i not in listened]
        valid_scores = valid_scores[:len(valid_cands)]

        # ── Genre Boost ───────────────────────────────────────────────────────
        valid_scores = self._apply_genre_boost(
            valid_cands, valid_scores,
            user_genre,
            self.cfg["genre_bonus_multiplier"],
        )

        # ── Churn Guard: Blend Trending khi churn cao ─────────────────────────
        if churn_risk >= self.cfg["churn_risk_threshold"]:
            tw = self.cfg["churn_trending_weight"]
            trending_max = float(self._trending_scores.max() + 1e-9)
            blended = []
            for iid, lgcn_sc in zip(valid_cands, valid_scores):
                t_sc = float(self._trending_scores[iid]) / trending_max
                blended.append((1.0 - tw) * lgcn_sc + tw * t_sc)
            valid_scores = blended

        # ── Hybrid TF-IDF Content ─────────────────────────────────────────────
        if content_alpha > 0 and self._content_index is not None:
            content_vec = self._get_user_content_vec(user_id_str)
            if content_vec is not None:
                hybrid_scores = []
                cold_pos_cache = {
                    iid: self._msid_to_cold_pos.get(self.idx2item[iid])
                    for iid in valid_cands
                }
                for iid, lgcn_sc in zip(valid_cands, valid_scores):
                    pos = cold_pos_cache.get(iid)
                    if pos is not None:
                        c_sc = float(self._cold_item_vecs[pos] @ content_vec[0])
                        score = (1 - content_alpha) * lgcn_sc + content_alpha * c_sc
                    else:
                        score = lgcn_sc
                    hybrid_scores.append(score)
                valid_scores = hybrid_scores

        # ── Artist Diversity Limit ────────────────────────────────────────────
        if artist_limit:
            artist_count = defaultdict(int)
            filtered_ids, filtered_scores = [], []
            for iid, sc in sorted(zip(valid_cands, valid_scores), key=lambda x: -x[1]):
                artist = self._iid_to_artist.get(iid, "")
                if artist_count[artist] < artist_limit:
                    filtered_ids.append(iid)
                    filtered_scores.append(sc)
                    artist_count[artist] += 1
                if len(filtered_ids) >= n * 3:
                    break
            valid_cands, valid_scores = filtered_ids, filtered_scores

        # ── MMR Reranking ─────────────────────────────────────────────────────
        final_ids, final_scores = self._mmr_rerank(
            valid_cands, valid_scores, n=n,
            lambda_=self.cfg["mmr_lambda"],
        )

        if not final_ids:
            return self.popular_items(n)

        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.insert(0, "user_id",    user_id_str)
        df.insert(1, "persona",    persona)
        df.insert(2, "churn_risk", f"{churn_risk:.1f}%")
        df.insert(3, "user_genre", user_genre or "—")
        df.index = range(1, len(df) + 1)
        return df

    def recommend(
        self,
        user_id_str: str,
        n: int = 10,
        seed_track_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """LightGCN thuần, không hybrid, không churn adjustment."""
        tier, uid = self._get_user_tier(user_id_str)
        if uid is None:
            return self.popular_items(n)
        base_vec = self.user_vectors[uid].copy()

        if seed_track_names:
            seed_vecs = []
            for tname in seed_track_names:
                t_lower = tname.lower()
                for msid, meta in self.item_meta.items():
                    if meta.get("track_name", "").lower() == t_lower and msid in self.item2idx:
                        seed_vecs.append(self.item_vectors[self.item2idx[msid]])
                        break
            if seed_vecs:
                seed_mean = np.mean(seed_vecs, axis=0)
                seed_norm = seed_mean / (np.linalg.norm(seed_mean) + 1e-8)
                base_vec  = 0.7 * base_vec + 0.3 * seed_norm

        query     = _normalize(base_vec)
        sc, idx   = self.index.search(query, n * 4)
        listened  = self._listened_set(user_id_str)
        rec_ids   = [int(i) for i in idx[0] if i not in listened]
        rec_scores= [float(s) for i, s in zip(idx[0], sc[0]) if i not in listened]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df) + 1)
        return df

    def recommend_cold_content(self, user_id_str: str, n: int = 10) -> pd.DataFrame:
        """Dành cho cold-start user: dùng genre preference từ DB."""
        user_info  = self._query_user(user_id_str)
        user_genre = user_info.get("dominant_genre", "POP")

        if user_genre:
            # Tìm bài thuộc genre này từ cold index
            genre_text = f"{user_genre} {user_genre}"
            cvec       = self._text_to_content_vec(genre_text)
            _, cpos    = self._cold_item_index.search(cvec, n * 5)
            results    = [self._to_row(self._cold_item_msids[p], 1.0, in_model=False)
                          for p in cpos[0][:n]]
        else:
            results = [self._to_row(self.idx2item[i], float(self._trending_scores[i]))
                       for i in np.argsort(self._trending_scores)[::-1][:n]]

        df = pd.DataFrame(results[:n])
        df.index = range(1, len(df) + 1)
        return df

    def recommend_trending(
        self,
        user_id_str: Optional[str] = None,
        n: int = 10,
        personal_weight: float = 0.5,
    ) -> pd.DataFrame:
        listened = self._listened_set(user_id_str) if user_id_str else set()
        tier, uid = self._get_user_tier(user_id_str) if user_id_str else ("cold", None)

        if uid is not None and personal_weight > 0:
            sc, idx = self.index.search(
                np.array(self.user_vectors[uid:uid+1], dtype=np.float32), n * 8
            )
            blend = {
                int(iid): (personal_weight * float(s) +
                           (1 - personal_weight) * float(self._trending_scores[iid]))
                for iid, s in zip(idx[0], sc[0]) if iid not in listened
            }
        else:
            blend = {int(i): float(self._trending_scores[i])
                     for i in np.argsort(self._trending_scores)[::-1][:n * 5]
                     if i not in listened}

        sorted_items = sorted(blend.items(), key=lambda x: x[1], reverse=True)
        cand_ids, cand_sc = [i for i, _ in sorted_items[:n * 3]], [s for _, s in sorted_items[:n * 3]]
        final_ids, final_scores = self._mmr_rerank(cand_ids, cand_sc, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df) + 1)
        return df

    def recommend_similar_to_new_item(
        self,
        track_name: str,
        artist_name: str,
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Tìm bài hát tương tự dựa trên TF-IDF (genre-enriched).
        Nếu nghệ sĩ có dominant_genre trong DB → nhồi thêm vào query.
        """
        genre     = self._query_artist_genre(artist_name) or ""
        # Nhân đôi artist & genre để tăng trọng số → text richer hơn bản gốc
        query_str = f"{artist_name} {artist_name} {track_name} {genre} {genre}".strip()
        cvec      = self._text_to_content_vec(query_str)

        _, cpos    = self._cold_item_index.search(cvec, n * 5)
        query_lower= f"{track_name} {artist_name}".lower()
        results    = []
        for pos in cpos[0]:
            msid = self._cold_item_msids[pos]
            meta = self.item_meta.get(msid, {})
            name = (meta.get("track_name", "") + " " + meta.get("artist_name", "")).lower().strip()
            if name == query_lower:
                continue
            results.append(self._to_row(msid, float(self._cold_item_vecs[pos] @ cvec[0]),
                                         in_model=(msid in self.item2idx)))
            if len(results) >= n:
                break

        df = pd.DataFrame(results)
        if not df.empty:
            df.insert(0, "similar_to", f"{track_name} ({artist_name})")
        df.index = range(1, len(df) + 1)
        return df

    def recommend_discovery(
        self,
        user_id_str: str,
        n: int = 10,
        serendipity: float = 0.3,
    ) -> pd.DataFrame:
        """Gợi ý 'thoát vùng an toàn': blend user vector với vector user xa lạ."""
        tier, uid = self._get_user_tier(user_id_str)
        if uid is None:
            return self.popular_items(n)
        self._ensure_user_index()
        _, far_nbs = self._user_index.search(
            np.array(self.user_vectors[uid:uid+1], dtype=np.float32),
            min(150, self.user_vectors.shape[0])
        )
        far_pool = far_nbs[0][min(50, len(far_nbs[0]) // 2):]
        far_vec  = (self.user_vectors[far_pool].mean(axis=0)
                    if len(far_pool) > 0
                    else np.random.default_rng(uid).standard_normal(self.dim).astype(np.float32))
        query    = _normalize(
            (1 - serendipity) * np.array(self.user_vectors[uid], dtype=np.float32)
            + serendipity * far_vec
        )
        sc, idx  = self.index.search(query, n * 6)
        listened = self._listened_set(user_id_str)
        rec_ids  = [int(i) for i in idx[0] if i not in listened]
        rec_sc   = [float(s) for i, s in zip(idx[0], sc[0]) if i not in listened]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_sc, n=n, lambda_=0.4)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df) + 1)
        return df

    def recommend_realtime(
        self,
        user_id_str: str,
        recent_listened_msids: List[str],
        n: int = 10,
        alpha: float = 0.4,
    ) -> pd.DataFrame:
        """Blend long-term user vector với short-term session vector."""
        tier, uid = self._get_user_tier(user_id_str)
        found     = [m for m in recent_listened_msids if m in self.item2idx]
        if not found:
            return self.recommend_hybrid(user_id_str, n=n)

        short_vec = np.mean([self.item_vectors[self.item2idx[m]] for m in found], axis=0)
        fused     = ((1 - alpha) * np.array(self.user_vectors[uid], dtype=np.float32) + alpha * short_vec
                     if uid is not None else short_vec)
        query     = _normalize(fused)
        sc, idx   = self.index.search(query, n * 4)
        exclude   = self._listened_set(user_id_str) | {self.item2idx[m] for m in found}
        rec_ids   = [int(i) for i in idx[0] if i not in exclude]
        rec_sc    = [float(s) for i, s in zip(idx[0], sc[0]) if i not in exclude]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_sc, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df) + 1)
        return df

    def recommend_next_in_session(
        self,
        session_msids: List[str],
        n: int = 10,
        decay: float = 0.8,
        filter_session: bool = True,
    ) -> pd.DataFrame:
        """Dự đoán bài hát tiếp theo trong session."""
        found = [(m, self.item2idx[m]) for m in session_msids if m in self.item2idx]
        if not found:
            return self.popular_items(n)
        n_f   = len(found)
        w     = np.array([decay ** (n_f - 1 - k) for k in range(n_f)], dtype=np.float32)
        query = self._proxy_vector_from_items(
            np.array([iid for _, iid in found], dtype=np.int32), weights=w / w.sum()
        )
        sc, idx    = self.index.search(query, n * 4)
        session_set= {iid for _, iid in found} if filter_session else set()
        rec_ids    = [int(i) for i in idx[0] if i not in session_set]
        rec_sc     = [float(s) for i, s in zip(idx[0], sc[0]) if i not in session_set]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_sc, n=n, lambda_=0.7)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df) + 1)
        return df

    def recommend_similar_users(
        self,
        user_id_str: str,
        n: int = 10,
        k_users: int = 20,
    ) -> pd.DataFrame:
        """Collaborative filtering qua user similarity."""
        tier, uid = self._get_user_tier(user_id_str)
        if uid is None:
            return self.popular_items(n)
        self._ensure_user_index()
        sim_sc, nbs = self._user_index.search(
            np.array(self.user_vectors[uid:uid+1], dtype=np.float32), k_users + 1
        )
        listened   = self._listened_set(user_id_str)
        item_score = defaultdict(float)
        for sim_uid, sc in zip(nbs[0], sim_sc[0]):
            if int(sim_uid) == uid:
                continue
            for iid in self.train_matrix[int(sim_uid)].indices:
                if iid not in listened:
                    item_score[int(iid)] += float(sc)
        if not item_score:
            return self.recommend_hybrid(user_id_str, n=n)
        sorted_items = sorted(item_score.items(), key=lambda x: x[1], reverse=True)
        cand_ids  = [i for i, _ in sorted_items[:n * 4]]
        cand_sc   = [s for _, s in sorted_items[:n * 4]]
        final_ids, final_scores = self._mmr_rerank(cand_ids, cand_sc, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df) + 1)
        return df

    def popular_items(self, n: int = 10) -> pd.DataFrame:
        top_ids = np.argsort(self._trending_scores)[::-1][:n].tolist()
        df = pd.DataFrame([self._to_row(self.idx2item[i], float(self._trending_scores[i]))
                           for i in top_ids])
        df.index = range(1, len(df) + 1)
        return df

    def get_user_profile(self, user_id_str: str) -> dict:
        """Trả về toàn bộ thông tin của một user từ Knowledge Base."""
        info = self._query_user(user_id_str)
        tier, uid = self._get_user_tier(user_id_str)
        info["model_tier"]     = tier
        info["n_interactions"] = int(self.train_matrix[uid].nnz) if uid is not None else 0
        return info

    # ── EVALUATION ────────────────────────────────────────────────────────────

    def evaluate_metrics(
        self,
        K: int = 20,
        eval_batch: int = 512,
        content_alpha: float = 0.0,
    ) -> dict:
        if self.test_matrix is None:
            print("  [SKIP] test_matrix không có")
            return {}
        from tqdm import tqdm
        test_users    = np.unique(self.test_matrix.nonzero()[0])
        n_users       = len(test_users)
        tot_rec = tot_pre = tot_ndcg = 0.0
        fetch_K = K * 4 if content_alpha > 0 else K * 2

        for start in tqdm(range(0, n_users, eval_batch), desc=f"Eval@{K}", unit="batch"):
            batch_u = test_users[start:start + eval_batch]
            sc_batch, idx_batch = self.index.search(
                np.array(self.user_vectors[batch_u], dtype=np.float32), fetch_K
            )
            for i, uid in enumerate(batch_u):
                actual     = set(self.test_matrix[uid].indices)
                if not actual:
                    continue
                train_seen = set(self.train_matrix[uid].indices.tolist())
                valid_mask = ~np.isin(idx_batch[i], list(train_seen))
                cands, scores = idx_batch[i][valid_mask], sc_batch[i][valid_mask]

                if content_alpha > 0:
                    cvec = self._get_user_content_vec(self.idx2user[uid])
                    if cvec is not None:
                        hybrid_sc = []
                        for iid, lgcn_sc in zip(cands, scores):
                            pos = self._msid_to_cold_pos.get(self.idx2item[iid])
                            hybrid_sc.append(
                                (1 - content_alpha) * lgcn_sc
                                + content_alpha * float(self._cold_item_vecs[pos] @ cvec[0])
                                if pos is not None else lgcn_sc
                            )
                        cands = cands[np.argsort(hybrid_sc)[::-1]]

                preds  = cands[:K]
                hits   = actual & set(preds)
                tot_rec  += len(hits) / len(actual)
                tot_pre  += len(hits) / K
                tot_ndcg += sum(
                    1.0 / math.log2(r + 2)
                    for r, p in enumerate(preds) if p in actual
                ) / sum(1.0 / math.log2(r + 2) for r in range(min(len(actual), K)))

        result = {
            f"Recall@{K}":    round(tot_rec / n_users, 4),
            f"Precision@{K}": round(tot_pre / n_users, 4),
            f"NDCG@{K}":      round(tot_ndcg / n_users, 4),
        }
        print(f"  {result}")
        return result

    def _ensure_user_index(self):
        if self._user_index is None:
            self._user_index = faiss.IndexFlatIP(self.dim)
            self._user_index.add(np.array(self.user_vectors, dtype=np.float32))

    def __del__(self):
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# QUICK DEMO
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rec = LocalRecommender()

    test_uid = rec.idx2user[0]
    profile  = rec.get_user_profile(test_uid)
    print(f"\n── User Profile: {test_uid}")
    for k, v in profile.items():
        print(f"   {k}: {v}")

    print("\n── [1] recommend_hybrid (Genre-Boost + Churn-Guard):")
    print(rec.recommend_hybrid(test_uid, n=5).to_string())

    print("\n── [2] recommend_trending:")
    print(rec.recommend_trending(n=5).to_string())

    print("\n── [3] recommend_similar_to_new_item ('Creep' by Radiohead):")
    print(rec.recommend_similar_to_new_item("Creep", "Radiohead", n=5).to_string())

    print("\n── [4] Evaluate Metrics:")
    rec.evaluate_metrics(K=20, content_alpha=0.0)
    rec.evaluate_metrics(K=20, content_alpha=0.25)

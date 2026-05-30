# =============================================================================
# CÀI ĐẶT MÔI TRƯỜNG & THƯ VIỆN
# =============================================================================
# !pip install scikit-learn scipy pandas numpy tqdm joblib faiss-cpu torch -q
import os, time, math, glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict, Counter
import joblib
from tqdm import tqdm
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

try:
    from IPython.display import display
except ImportError:
    display = print

import warnings
warnings.filterwarnings('ignore')
print('✅ Thư viện đã sẵn sàng.')

# =============================================================================
# LỚP RECOMMENDER HOÀN CHỈNH (TÍCH HỢP TẤT CẢ)
# =============================================================================
import gc

class AdvancedHybridRecommender:
    def __init__(self, model_dir: str, cache_dir: str = './cache/', cold_threshold: int = 5):
        t0 = time.time()
        print(f'[Recommender] Loading artifacts từ {model_dir}...')
        self.model_dir      = model_dir
        self.cache_dir      = cache_dir
        self.cold_threshold = cold_threshold
        #os.makedirs(self.cache_dir, exist_ok=True)

        # 1. Embeddings (Sử dụng mmap_mode='r' để ánh xạ từ ổ cứng, chống tràn RAM)
        self.user_vectors = np.load(os.path.join(model_dir, 'user_vectors.npy'), mmap_mode='r')
        self.item_vectors = np.load(os.path.join(model_dir, 'item_vectors.npy'), mmap_mode='r')
        self.dim = self.item_vectors.shape[1]

        # 2. Mappings
        m = joblib.load(os.path.join(model_dir, 'index_mappings.pkl'))
        self.user2idx            = m['user2idx']
        self.item2idx            = m['item2idx']
        self.idx2user            = m['idx2user']
        self.idx2item            = m['idx2item']
        self.item_meta           = m['item_meta']
        self.user_raw_items      = m['user_raw_items']
        self.user_item_ts_matrix = m.get('user_item_ts_matrix', {})
        self.global_max_ts       = m.get('global_max_ts', 0.0)

        # Xóa Dictionary 'm' ngay lập tức và ép thu gom rác giải phóng RAM
        del m
        gc.collect()

        # 3. Sparse matrices
        self.train_matrix = sp.load_npz(os.path.join(model_dir, 'train_user_item.npz'))
        self.test_matrix  = sp.load_npz(os.path.join(model_dir, 'test_user_item.npz'))

        # 4. LightGCN FAISS index
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.item_vectors)
        self._user_index = None

        print(f'  ✓ user={self.user_vectors.shape} | item={self.item_vectors.shape}')
        print(f'  ✓ {len(self.user2idx):,} users | {len(self.item2idx):,} warm items')

        # 5. Content & auxiliary indexes - SỬ DỤNG MASTER CACHE
        self._load_or_build_master_cache()

        print(f'✅ Recommender sẵn sàng ({time.time()-t0:.1f}s)')

    def update_tfidf_index(self, force_rebuild=False):
        """Hàm cập nhật lại TF-IDF nếu có dữ liệu mới."""
        print("\n[Update TF-IDF] Đang kiểm tra và cập nhật...")
        self._load_or_build_master_cache(force_rebuild=force_rebuild)
        print("[Update TF-IDF] Hoàn tất!")

    # ─────────────────────────────────────────────────────────────────────────
    # A. MASTER CACHE HANDLER
    # ─────────────────────────────────────────────────────────────────────────
    def _load_or_build_master_cache(self, force_rebuild=False):
        master_cache_path = os.path.join(self.cache_dir, 'master_indexes_cache.pkl')

        tfidf_path  = os.path.join(self.cache_dir, f'tfidf_model.pkl')
        svd_path    = os.path.join(self.cache_dir, f'svd_64_model.pkl')
        
        if not force_rebuild and os.path.exists(master_cache_path):
            print('  [Cache] Đang load toàn bộ Master Cache (bỏ qua bước build loop)...')
            tc = time.time()
            
            # Dùng mmap_mode='r' để các Numpy Arrays nội bộ trong Pickle không ngốn thêm RAM
            cached_data = joblib.load(master_cache_path, mmap_mode='r')
            
            # Cập nhật toàn bộ thuộc tính đã lưu vào class
            for key, value in cached_data.items():
                setattr(self, key, value)
            
            # Khôi phục FAISS Indexes từ các vectors numpy (cực nhanh, chỉ ~0.01s)
            self._rebuild_faiss_indexes()

            # Thêm 2 dòng này để load mô hình TF-IDF và SVD
            self._tfidf = joblib.load(tfidf_path)
            self._svd   = joblib.load(svd_path)
            
            # Hủy biến dict tạm và ép thu gom rác để xóa bỏ phần RAM bị nhân đôi
            del cached_data
            gc.collect()
            
            print(f'  [Cache] Load Master Cache thành công! ({time.time()-tc:.2f}s)')
        else:
            print('  [Cache] Không tìm thấy Master Cache hoặc yêu cầu rebuild. Đang build từ đầu...')
            self._build_content_index(force_rebuild=force_rebuild)
            self._build_artist_index()
            self._build_trending_scores()
            
            # Lưu các kết quả (dictionary, numpy arrays) vào Master Cache
            print('  [Cache] Đang lưu dữ liệu vào Master Cache để dùng cho lần sau...')
            data_to_cache = {
                # Content properties
                '_content_dim': getattr(self, '_content_dim', 64),
                '_cold_item_msids': self._cold_item_msids,
                '_cold_item_vecs': self._cold_item_vecs,
                '_msid_to_cold_pos': self._msid_to_cold_pos,
                '_content_iids': self._content_iids,
                '_content_vecs': self._content_vecs,
                '_iid_to_content_pos': self._iid_to_content_pos,
                
                # Artist properties
                '_iid_to_artist': self._iid_to_artist,
                '_artist2items': self._artist2items,
                '_artist_names': self._artist_names,
                '_artist_name_to_pos': self._artist_name_to_pos,
                '_artist_vecs': self._artist_vecs,
                
                # Trending properties
                '_trending_scores': self._trending_scores
            }
            joblib.dump(data_to_cache, master_cache_path)
            
            # Dọn RAM sau khi dump cache thành công
            del data_to_cache
            gc.collect()

    def _rebuild_faiss_indexes(self):
        """Khôi phục lại các đối tượng FAISS vì chúng không nên được pickle trực tiếp"""
        # Cold index
        self._cold_item_index = faiss.IndexFlatIP(self._content_dim)
        self._cold_item_index.add(self._cold_item_vecs)
        
        # Content warm index
        self._content_index = faiss.IndexFlatIP(self._content_dim)
        self._content_index.add(self._content_vecs)
        
        # Artist index
        self._artist_index = faiss.IndexFlatIP(self.dim)
        self._artist_index.add(self._artist_vecs)

    # ─────────────────────────────────────────────────────────────────────────
    # B. BUILD INDEXES (GIỮ NGUYÊN LOGIC CỦA BẠN, CHỈ ĐƯỢC GỌI KHI KHÔNG CÓ MASTER CACHE)
    # ─────────────────────────────────────────────────────────────────────────
    def _build_content_index(self, svd_dim: int = 64, cold_max_features: int = 30_000, force_rebuild=False):
        t0 = time.time()
        valid_msids = set()
        for raw_set in self.user_raw_items.values():
            valid_msids.update(raw_set)

        # BỘ LỌC CHỐNG SPAM
        forbidden = {'', 'unknown', 'various artists', 'traditional', 'various', 'unknown artist'}
        known_artists = set()
        for msid in self.item2idx:
            a = self.item_meta.get(msid, {}).get('artist_name', '').strip().lower()
            if a and a not in forbidden:
                known_artists.add(a)

        all_msids, all_texts = [], []
        artist_track_count = defaultdict(int)
        MAX_COLD_PER_ARTIST = 50

        print('  [Content] Đang build corpus (Text: artist x2 + track)...')
        for msid, meta in tqdm(self.item_meta.items(), desc='Xây dựng Corpus TF-IDF', unit=' bài'):
            artist = meta.get('artist_name', '').strip().lower()
            is_warm         = msid in self.item2idx
            has_interaction = msid in valid_msids

            if is_warm or has_interaction:
                keep = True
            elif artist in known_artists and artist_track_count[artist] < MAX_COLD_PER_ARTIST:
                keep = True
                artist_track_count[artist] += 1
            else:
                keep = False

            if keep:
                track = meta.get('track_name', '').strip().lower()
                text = f"{artist} {artist} {track}".strip()
                all_msids.append(msid)
                all_texts.append(text if text else 'unknown')

        corpus_size = len(all_texts)
        
        tfidf_path  = os.path.join(self.cache_dir, f'tfidf_model.pkl')
        svd_path    = os.path.join(self.cache_dir, f'svd_64_model.pkl')

        # TÁI SỬ DỤNG CACHE TF-IDF/SVD
        if not force_rebuild and os.path.exists(tfidf_path) and os.path.exists(svd_path):
            print(f'  [Content] Load mô hình TF-IDF/SVD...')
            self._tfidf = joblib.load(tfidf_path)
            self._svd   = joblib.load(svd_path)
            
            # SỬA Ở ĐÂY: Chia batch để transform chống OOM
            batch_size = 500_000 
            dense_list = []
            for i in tqdm(range(0, corpus_size, batch_size), desc='Batch Transform'):
                batch_texts = all_texts[i:i+batch_size]
                batch_tfidf = self._tfidf.transform(batch_texts)
                batch_dense = self._svd.transform(batch_tfidf).astype(np.float32)
                dense_list.append(batch_dense)
                del batch_texts, batch_tfidf, batch_dense
                gc.collect()
            
            dense_all = np.vstack(dense_list)
            real_dim  = self._svd.n_components
            del dense_list, all_texts
            gc.collect()
        else:
            print(f'  [Content] Fit TF-IDF/SVD mới ({corpus_size:,} items)...')
            if force_rebuild:
                for f in glob.glob(os.path.join(self.cache_dir, 'tfidf_*.pkl')) + glob.glob(os.path.join(self.cache_dir, 'svd_*.pkl')):
                    try: os.remove(f)
                    except: pass
            
            self._tfidf = TfidfVectorizer(
                analyzer='word', ngram_range=(1, 1),
                max_features=20000,
                sublinear_tf=True, min_df=5,
            )
            tfidf_all = self._tfidf.fit_transform(all_texts)
            
            # SỬA Ở ĐÂY: Xóa mảng all_texts sớm để lấy lại RAM
            del all_texts
            gc.collect()
            
            real_dim  = min(svd_dim, tfidf_all.shape[1] - 1, tfidf_all.shape[0] - 1)
            self._svd = TruncatedSVD(n_components=real_dim, random_state=42, n_iter=5)
            dense_all = self._svd.fit_transform(tfidf_all).astype(np.float32)
            joblib.dump(self._tfidf, tfidf_path)
            joblib.dump(self._svd,   svd_path)
            
            # SỬA Ở ĐÂY: Xóa biến trung gian
            del tfidf_all
            gc.collect()

        faiss.normalize_L2(dense_all)
        self._content_dim = real_dim

        # 1. COLD INDEX (TOÀN BỘ)
        self._cold_item_msids  = np.array(all_msids)
        self._cold_item_vecs   = dense_all
        self._cold_item_index  = faiss.IndexFlatIP(real_dim)
        self._cold_item_index.add(dense_all)
        self._msid_to_cold_pos = {msid: pos for pos, msid in enumerate(all_msids)}

        # 2. WARM INDEX (CHỈ TRONG MODEL)
        warm_iids, warm_rows = [], []
        for msid, iid in tqdm(self.item2idx.items(), desc='Trích xuất Warm Items', unit=' bài'):
            pos = self._msid_to_cold_pos.get(msid)
            if pos is not None:
                warm_iids.append(iid)
                warm_rows.append(pos)

        warm_vecs = dense_all[warm_rows].copy()
        self._content_iids       = np.array(warm_iids, dtype=np.int32)
        self._content_vecs       = warm_vecs
        self._content_index      = faiss.IndexFlatIP(real_dim)
        self._content_index.add(warm_vecs)
        self._iid_to_content_pos = {int(iid): pos for pos, iid in enumerate(warm_iids)}

        print(f'  [Content] SVD({real_dim}), corpus={corpus_size:,}, warm={len(warm_iids):,} | {time.time()-t0:.1f}s')

    def _build_artist_index(self):
        t0 = time.time()
        artist2items = defaultdict(list)
        self._iid_to_artist = {}
        for msid, meta in tqdm(self.item_meta.items(), desc='Ánh xạ Nghệ sĩ', unit=' bài'):
            if msid in self.item2idx:
                artist = meta.get('artist_name', '').strip().lower()
                iid    = self.item2idx[msid]
                if artist:
                    artist2items[artist].append(iid)
                    self._iid_to_artist[iid] = artist
        self._artist2items       = dict(artist2items)
        self._artist_names       = list(artist2items.keys())
        self._artist_name_to_pos = {n: i for i, n in enumerate(self._artist_names)}
        artist_vecs = np.array([self.item_vectors[items].mean(axis=0) for items in artist2items.values()], dtype=np.float32)
        faiss.normalize_L2(artist_vecs)
        self._artist_vecs  = artist_vecs
        self._artist_index = faiss.IndexFlatIP(self.dim)
        self._artist_index.add(artist_vecs)
        print(f'  [Artist] {len(self._artist2items):,} artists | {time.time()-t0:.1f}s')

    def _build_trending_scores(self, halflife_days: int = 30):
        t0  = time.time()
        n   = self.item_vectors.shape[0]
        pop = np.asarray(self.train_matrix.sum(axis=0)).flatten().astype(np.float64)
        ts_sum, ts_count = np.zeros(n, np.float64), np.zeros(n, np.int32)
        for ts_dict in tqdm(self.user_item_ts_matrix.values(), desc='Tính Trending', unit=' user'):
            for iid, ts in ts_dict.items():
                i = int(iid)
                if 0 <= i < n:
                    ts_sum[i] += float(ts); ts_count[i] += 1
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_ts = np.where(ts_count > 0, ts_sum / np.maximum(ts_count, 1), float(self.global_max_ts))
        hl_sec   = halflife_days * 86400.0
        days_ago = np.clip((self.global_max_ts - avg_ts) / (hl_sec + 1e-9), 0, None)
        trending = pop * np.exp(-days_ago)
        t_max    = trending.max()
        self._trending_scores = (trending / t_max).astype(np.float32) if t_max > 0 else trending.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # B. HELPERS (TỪ V2)
    # ─────────────────────────────────────────────────────────────────────────

    def _to_row(self, msid, score, in_model=True):
        meta = self.item_meta.get(msid, {})
        return {'track_name' : meta.get('track_name', msid), 'artist_name': meta.get('artist_name', ''),
                'score'      : round(float(score), 4), 'in_model'   : in_model}

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        v = vec.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def _listened_set(self, user_id_str):
        raw = self.user_raw_items.get(str(user_id_str), set())
        return {self.item2idx[m] for m in raw if m in self.item2idx}

    def _get_user_tier(self, user_id_str: str):
        uid = self.user2idx.get(str(user_id_str))
        if uid is None: return 'cold', None
        return ('warm' if self.train_matrix[uid].nnz >= self.cold_threshold else 'lukewarm'), uid

    def _proxy_vector_from_items(self, iids, weights=None) -> np.ndarray:
        iids = np.asarray(iids, dtype=np.int32)
        vecs = self.item_vectors[iids].astype(np.float32)
        if weights is not None:
            w = np.asarray(weights, np.float32); w /= w.sum() + 1e-12
            vec = (vecs * w[:, None]).sum(axis=0, keepdims=True)
        else: vec = vecs.mean(axis=0, keepdims=True)
        return self._normalize(vec)

    def _text_to_content_vec(self, text: str) -> np.ndarray:
        v = self._svd.transform(self._tfidf.transform([text.strip().lower()])).astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def _get_user_content_vec(self, user_id_str: str):
        raw = self.user_raw_items.get(str(user_id_str), set())
        if not raw: return None
        uid = self.user2idx.get(str(user_id_str))
        if uid is not None:
            ts_dict = self.user_item_ts_matrix.get(uid, {})
            if ts_dict:
                sorted_iids = sorted(ts_dict.keys(), key=lambda i: ts_dict[i], reverse=True)[:100]
                history_msids = [self.idx2item[i] for i in sorted_iids if i < len(self.idx2item)]
            else: history_msids = [self.idx2item[i] for i in self.train_matrix[uid].indices]
        else: history_msids = sorted(raw)[:100]

        vecs = [self._cold_item_vecs[self._msid_to_cold_pos[m]] for m in history_msids if self._msid_to_cold_pos.get(m) is not None]
        if not vecs: return None
        v = np.mean(vecs, axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(v)
        return v

    def _apply_artist_diversity(self, rec_ids, rec_scores, artist_limit=2):
        artist_count, f_ids, f_scores = defaultdict(int), [], []
        for iid, sc in zip(rec_ids, rec_scores):
            artist = self._iid_to_artist.get(int(iid), f'__unk_{iid}')
            if artist_count[artist] < artist_limit:
                f_ids.append(iid); f_scores.append(sc)
                artist_count[artist] += 1
        return f_ids, f_scores

    def _mmr_rerank(self, rec_ids, scores, n=10, lambda_=0.6):
        if not rec_ids: return [], []
        rec_ids, scores = np.array(rec_ids), np.array(scores)
        norm_sc = scores / (scores.max() + 1e-8)
        factors = self.item_vectors[rec_ids]
        sim_mat = factors @ factors.T
        selected, unsel = [], list(range(len(rec_ids)))
        first = int(np.argmax(norm_sc))
        selected.append(first); unsel.remove(first)
        while len(selected) < n and unsel:
            rel = norm_sc[unsel]
            sim = sim_mat[np.ix_(unsel, selected)].max(axis=1)
            mmr_scores = lambda_ * rel - (1 - lambda_) * sim
            best = int(np.argmax(mmr_scores))
            selected.append(unsel[best]); unsel.pop(best)
        return rec_ids[selected].tolist(), scores[selected].tolist()

    def _cold_user_vector(self, liked_tracks=None, liked_artists=None, user_id_str=None):
        proxy_iids = []
        if liked_tracks:
            for track in liked_tracks:
                cvec = self._text_to_content_vec(track)
                _, cpos = self._content_index.search(cvec, 5)
                proxy_iids.extend(self._content_iids[cpos[0]].tolist())
        if liked_artists:
            for artist in liked_artists:
                a_lower = artist.strip().lower()
                if a_lower in self._artist2items: proxy_iids.extend(self._artist2items[a_lower][:10])
                else:
                    cvec = self._text_to_content_vec(a_lower)
                    _, cpos = self._content_index.search(cvec, 5)
                    proxy_iids.extend(self._content_iids[cpos[0]].tolist())
        if user_id_str and not proxy_iids:
            raw = self.user_raw_items.get(str(user_id_str), set())
            proxy_iids.extend(self.item2idx[m] for m in raw if m in self.item2idx)
        if not proxy_iids: return None
        return self._proxy_vector_from_items(list(dict.fromkeys(proxy_iids)))

    def popular_items(self, n=10):
        counts  = np.asarray(self.train_matrix.sum(axis=0)).flatten()
        top_ids = np.argsort(counts)[::-1][:n]
        df = pd.DataFrame([self._to_row(self.idx2item[i], counts[i]) for i in top_ids])
        df.index = range(1, len(df)+1)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # C. GỢI Ý ĐẦY ĐỦ (V2 + BẢN GỐC)
    # ─────────────────────────────────────────────────────────────────────────

    def recommend(self, user_id_str, n=10, filter_listened=True, use_mmr=True):
        uid = self.user2idx.get(str(user_id_str))
        if uid is None:
            raw_msids = self.user_raw_items.get(str(user_id_str), set())
            if len(raw_msids) < 3: return self.popular_items(n)
            listened_iids = [self.item2idx[m] for m in raw_msids if m in self.item2idx]
            user_vec = self._normalize(np.mean(self.item_vectors[listened_iids], axis=0))
        else: user_vec = self.user_vectors[uid:uid+1]
        scores, indices = self.index.search(user_vec, n * 5)
        listened = self._listened_set(user_id_str)
        rec_ids, rec_scores = [int(i) for i in indices[0] if not filter_listened or i not in listened], [float(s) for i, s in zip(indices[0], scores[0]) if not filter_listened or i not in listened]
        if use_mmr: rec_ids, rec_scores = self._mmr_rerank(rec_ids, rec_scores, n=n)
        else: rec_ids, rec_scores = rec_ids[:n], rec_scores[:n]
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(rec_ids, rec_scores)])
        df.index = range(1, len(df)+1)
        return df

    def recommend_hybrid(self, user_id_str, n=10, liked_tracks=None, liked_artists=None, filter_listened=True, use_mmr=True, artist_limit=2, trending_boost=0.0, content_alpha=0.25, auto_refresh=False, exclude_history=True):
        tier, uid = self._get_user_tier(user_id_str)
        if tier == 'warm': query_vec = self.user_vectors[uid:uid+1].copy()
        elif tier == 'lukewarm':
            n_inter  = self.train_matrix[uid].nnz
            lgcn_vec = self.user_vectors[uid].copy()
            proxy_vec = self._proxy_vector_from_items(list(self.train_matrix[uid].indices)).flatten()
            alpha = min(0.5 + 0.5 * (n_inter / max(self.cold_threshold, 1)), 1.0)
            query_vec = self._normalize(alpha * lgcn_vec + (1.0 - alpha) * proxy_vec)
        else:
            query_vec = self._cold_user_vector(liked_tracks, liked_artists, user_id_str)
            if query_vec is None: return self.popular_items(n)

        lgcn_scores, indices = self.index.search(query_vec, n * 8)
        listened = self._listened_set(user_id_str) if filter_listened else set()
        content_vec = self._get_user_content_vec(user_id_str) if content_alpha > 0 else None

        rec_ids, rec_scores = [], []
        for iid, lgcn_sc in zip(indices[0], lgcn_scores[0]):
            if filter_listened and iid in listened: continue
            iid = int(iid); lgcn_sc = float(lgcn_sc)
            if trending_boost > 0.0: lgcn_sc = (1.0 - trending_boost) * lgcn_sc + trending_boost * float(self._trending_scores[iid])
            if content_vec is not None:
                msid = self.idx2item[iid]
                pos  = self._msid_to_cold_pos.get(msid)
                hybrid_sc  = (1.0 - content_alpha) * lgcn_sc + content_alpha * float(self._cold_item_vecs[pos] @ content_vec[0]) if pos is not None else lgcn_sc
            else: hybrid_sc = lgcn_sc
            rec_ids.append(iid); rec_scores.append(hybrid_sc)

        if artist_limit > 0: rec_ids, rec_scores = self._apply_artist_diversity(rec_ids, rec_scores, artist_limit)
        pool_n = n * 3 if auto_refresh else n
        if use_mmr and len(rec_ids) >= pool_n:
            final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=pool_n)
        else:
            final_ids, final_scores = rec_ids[:pool_n], rec_scores[:pool_n]

        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.insert(3, 'hybrid_alpha', content_alpha)
        df.index = range(1, len(df) + 1)

        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=user_id_str,
                                             exclude_history=exclude_history)
        return df

    def recommend_inclusive(self, user_id_str, n_warm: int = 7, n_cold: int = 3, auto_refresh: bool = False, exclude_history: bool = True) -> pd.DataFrame:
        print(f'\n[inclusive] User={user_id_str!r} | {n_warm} Warm + {n_cold} Cold')
        warm_df = self.recommend(user_id_str, n=n_warm, filter_listened=True, use_mmr=True)
        if not warm_df.empty: warm_df['source'] = 'LightGCN (Top Hit)'

        raw_history = self.user_raw_items.get(str(user_id_str), set())
        if not raw_history: return warm_df

        uid = self.user2idx.get(str(user_id_str))
        if uid is not None and self.user_item_ts_matrix.get(uid):
            ts_dict  = self.user_item_ts_matrix[uid]
            sorted_i = sorted(ts_dict.keys(), key=lambda i: ts_dict[i], reverse=True)[:50]
            hist_msids = [self.idx2item[i] for i in sorted_i if i < len(self.idx2item)]
        else: hist_msids = sorted(raw_history)[:50]

        vecs = [self._cold_item_vecs[self._msid_to_cold_pos[m]] for m in hist_msids if self._msid_to_cold_pos.get(m) is not None]
        if not vecs: return warm_df

        user_tfidf_vec = np.mean(vecs, axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(user_tfidf_vec)
        scores, cand_pos = self._cold_item_index.search(user_tfidf_vec, n_cold * 10)

        cold_records = []
        for pos, sc in zip(cand_pos[0], scores[0]):
            msid = self._cold_item_msids[pos]
            if msid not in self.item2idx and msid not in raw_history:
                meta = self.item_meta.get(msid, {})
                cold_records.append({'track_name': meta.get('track_name', msid), 'artist_name': meta.get('artist_name', ''), 'score': round(float(sc), 4), 'in_model': False, 'source': 'TF-IDF (Hidden Gem)'})
            if len(cold_records) >= n_cold: break

        cold_df = pd.DataFrame(cold_records)
        if cold_df.empty: return warm_df
        final_df = pd.concat([warm_df, cold_df], ignore_index=True)
        final_df.index = range(1, len(final_df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(final_df, n=(n_warm + n_cold), user_id_str=user_id_str, exclude_history=exclude_history)
        return final_df

    def recommend_cold_content(self, user_id_str=None, text_query=None, n=10, auto_refresh: bool = False, exclude_history: bool = True):
        if text_query:
            query_vec = self._text_to_content_vec(text_query)
            source    = f'query="{text_query}"'
        elif user_id_str:
            query_vec = self._get_user_content_vec(user_id_str)
            if query_vec is None: return self.popular_items(n)
            source = f'user={user_id_str}'
        else: return self.popular_items(n)

        raw_history = self.user_raw_items.get(str(user_id_str), set()) if user_id_str else set()
        scores, cand_pos = self._cold_item_index.search(query_vec, n * 5)
        
        records = []
        for pos, sc in zip(cand_pos[0], scores[0]):
            msid = self._cold_item_msids[pos]
            if msid in raw_history: continue
            meta = self.item_meta.get(msid, {})
            records.append({'track_name': meta.get('track_name', msid), 'artist_name': meta.get('artist_name', ''), 'score': round(float(sc), 4), 'in_model': msid in self.item2idx, 'source': f'TF-IDF ({source})'})
            if len(records) >= n: break
        df = pd.DataFrame(records)
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=user_id_str, exclude_history=exclude_history)
        return df

    def generate_playlist(self, user_id_str, seed_track_names=None, playlist_name='Gợi ý dành riêng cho bạn', n_songs=15, auto_refresh: bool = False, exclude_history: bool = True):
        print(f'\n[playlist] "{playlist_name}"')
        uid = self.user2idx.get(str(user_id_str))
        if uid is None: return self.popular_items(n_songs)

        base_vec = self.user_vectors[uid].copy()
        if seed_track_names:
            seed_vecs = []
            for track in seed_track_names:
                t_lower = track.lower()
                for msid, meta in self.item_meta.items():
                    if meta.get('track_name','').lower() == t_lower and msid in self.item2idx:
                        seed_vecs.append(self.item_vectors[self.item2idx[msid]])
                        break
            if seed_vecs:
                seed_mean = np.mean(seed_vecs, axis=0)
                seed_norm = seed_mean / (np.linalg.norm(seed_mean) + 1e-8)
                base_vec  = 0.7 * base_vec + 0.3 * seed_norm
                print(f'  -> Blend: 70% user profile + 30% seed ({len(seed_vecs)} tracks)')

        query_vec = self._normalize(base_vec)
        scores, indices = self.index.search(query_vec, n_songs * 4)
        listened = self._listened_set(user_id_str)
        rec_ids, rec_scores = [int(i) for i in indices[0] if i not in listened], [float(s) for i, s in zip(indices[0], scores[0]) if i not in listened]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n_songs, lambda_=0.5)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)]) 
        df = df.drop_duplicates(subset=["track_name", "artist_name"])
        df = df.head(n_songs)
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n_songs, user_id_str=user_id_str, exclude_history=exclude_history)
        return df

    def recommend_realtime(self, user_id_str, recent_listened_msids, n=10, alpha=0.4, auto_refresh: bool = False, exclude_history: bool = True):
        print(f'\n[realtime] User {user_id_str} | α={alpha}')
        uid = self.user2idx.get(str(user_id_str))
        found_msids = [m for m in recent_listened_msids if m in self.item2idx]
        if not found_msids: return self.recommend(user_id_str, n=n)

        short_vec = np.mean([self.item_vectors[self.item2idx[m]] for m in found_msids], axis=0)
        fused = (1 - alpha) * self.user_vectors[uid] + alpha * short_vec if uid is not None else short_vec

        query_vec = self._normalize(fused)
        scores, indices = self.index.search(query_vec, n * 4)
        exclude = self._listened_set(user_id_str) | {self.item2idx[m] for m in found_msids}
        rec_ids, rec_scores = [int(i) for i in indices[0] if i not in exclude], [float(s) for i, s in zip(indices[0], scores[0]) if i not in exclude]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=user_id_str, exclude_history=exclude_history)
        return df

    def recommend_by_timeframe(self, user_id_str, start_date, end_date, n=10, auto_refresh: bool = False, exclude_history: bool = True):
        print(f'\n[timeframe] {start_date} → {end_date}')
        uid = self.user2idx.get(str(user_id_str))
        if uid is None: return self.popular_items(n)
        start_ts, end_ts = pd.Timestamp(start_date).timestamp(), pd.Timestamp(end_date).timestamp()
        ts_dict  = self.user_item_ts_matrix.get(uid, {})
        period_iids = [iid for iid, ts in ts_dict.items() if start_ts <= ts <= end_ts]
        if not period_iids: return pd.DataFrame()

        ts_arr = np.array([ts_dict[iid] for iid in period_iids])
        w = np.exp(-(ts_arr.max() - ts_arr) / (30 * 86400))
        period_vec = self._normalize((self.item_vectors[period_iids] * (w/w.sum())[:, None]).sum(axis=0))

        scores, indices = self.index.search(period_vec, n * 3)
        period_set = set(period_iids)
        rec_ids, rec_scores = [int(i) for i in indices[0] if i not in period_set], [float(s) for i, s in zip(indices[0], scores[0]) if i not in period_set]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=user_id_str, exclude_history=exclude_history)
        return df

    def recommend_similar_to_new_item(self, track_name, artist_name, n=10, include_cold_items=True, auto_refresh: bool = False, exclude_history: bool = True):
        query = f'{artist_name} {artist_name} {track_name}'
        cvec  = self._text_to_content_vec(query)
        if include_cold_items:
            n_search = min(n * 5, self._cold_item_index.ntotal)
            _, cpos  = self._cold_item_index.search(cvec, n_search)
            results, query_lower = [], f'{track_name} {artist_name}'.lower()
            for pos in cpos[0]:
                msid = self._cold_item_msids[pos]
                meta = self.item_meta.get(msid, {})
                name = (meta.get('track_name','') + ' ' + meta.get('artist_name','')).strip().lower()
                if name == query_lower: continue
                results.append(self._to_row(msid, float(self._cold_item_vecs[pos] @ cvec[0]), in_model=(msid in self.item2idx)))
                if len(results) >= n: break
            df = pd.DataFrame(results)
        else:
            n_search = min(30, len(self._content_iids))
            _, cpos  = self._content_index.search(cvec, n_search)
            cand_iids = self._content_iids[cpos[0]]
            if not len(cand_iids): return pd.DataFrame()
            proxy_vec  = self._proxy_vector_from_items(cand_iids[:15])
            scores, indices = self.index.search(proxy_vec, n * 3)
            query_lower = f'{track_name} {artist_name}'.lower()
            final_ids, final_scores = [], []
            for iid, sc in zip(indices[0], scores[0]):
                msid = self.idx2item[iid]
                meta = self.item_meta.get(msid, {})
                name = (meta.get('track_name','') + ' ' + meta.get('artist_name','')).strip().lower()
                if name == query_lower: continue
                final_ids.append(int(iid)); final_scores.append(float(sc))
                if len(final_ids) >= n: break
            df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])

        if not df.empty:
            df.insert(0, 'similar_to', f'{track_name} ({artist_name})')
            df.index = range(1, len(df)+1)
            if auto_refresh:
                return self._apply_refresh_logic(df, n=n, user_id_str=None, exclude_history=exclude_history)
        return df

    def recommend_trending(self, user_id_str=None, n=10, personal_weight=0.5, auto_refresh: bool = False, exclude_history: bool = True):
        listened = self._listened_set(user_id_str) if user_id_str else set()
        if user_id_str is None or personal_weight == 0.0:
            top_ids = [int(i) for i in np.argsort(self._trending_scores)[::-1] if i not in listened][:n]
            df = pd.DataFrame([{**self._to_row(self.idx2item[i], self._trending_scores[i]), 'trending_rank': r+1} for r, i in enumerate(top_ids)])
            df.index = range(1, len(df)+1)
            return df
        tier, uid = self._get_user_tier(user_id_str)
        if uid is not None:
            scores, indices = self.index.search(self.user_vectors[uid:uid+1], n*8)
            blend = {int(iid): (personal_weight*float(sc) + (1.0-personal_weight)*float(self._trending_scores[iid])) for iid, sc in zip(indices[0], scores[0]) if iid not in listened}
        else: blend = {int(i): float(self._trending_scores[i]) for i in np.argsort(self._trending_scores)[::-1][:n*5] if i not in listened}

        sorted_blend = sorted(blend.items(), key=lambda x: x[1], reverse=True)
        cand_ids, cand_sc = [i for i,_ in sorted_blend[:n*3]], [s for _,s in sorted_blend[:n*3]]
        final_ids, final_scores = self._mmr_rerank(cand_ids, cand_sc, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=user_id_str, exclude_history=exclude_history)
        return df

    def _ensure_user_index(self):
        if self._user_index is None:
            self._user_index = faiss.IndexFlatIP(self.dim)
            self._user_index.add(self.user_vectors)

    def recommend_similar_users(self, user_id_str, n=10, k_users=20, filter_listened=True, auto_refresh: bool = False, exclude_history: bool = True):
        tier, uid = self._get_user_tier(user_id_str)
        if uid is None: return self.popular_items(n)
        self._ensure_user_index()
        sim_scores, nbs = self._user_index.search(self.user_vectors[uid:uid+1], k_users+1)
        listened = self._listened_set(user_id_str) if filter_listened else set()
        item_score = defaultdict(float)
        for sim_uid, sim_sc in zip(nbs[0], sim_scores[0]):
            if int(sim_uid) == uid: continue
            for iid in self.train_matrix[int(sim_uid)].indices:
                if iid not in listened: item_score[int(iid)] += float(sim_sc)
        if not item_score: return self.recommend_hybrid(user_id_str, n=n)
        sorted_items = sorted(item_score.items(), key=lambda x: x[1], reverse=True)
        cand_ids, cand_sc = [i for i,_ in sorted_items[:n*4]], [s for _,s in sorted_items[:n*4]]
        final_ids, final_scores = self._mmr_rerank(cand_ids, cand_sc, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=user_id_str, exclude_history=exclude_history)
        return df

    def recommend_discovery(self, user_id_str, n=10, serendipity=0.3, auto_refresh: bool = False, exclude_history: bool = True):
        tier, uid = self._get_user_tier(user_id_str)
        if uid is None: return self.popular_items(n)
        self._ensure_user_index()
        _, far_nbs = self._user_index.search(self.user_vectors[uid:uid+1], min(150, self.user_vectors.shape[0]))
        far_pool = far_nbs[0][min(50, len(far_nbs[0])//2):]
        far_vec  = self.user_vectors[far_pool].mean(axis=0) if len(far_pool) > 0 else (np.random.default_rng(uid).standard_normal(self.dim).astype(np.float32))
        query_vec = self._normalize((1.0-serendipity)*self.user_vectors[uid].copy() + serendipity*far_vec)
        scores, indices = self.index.search(query_vec, n*6)
        listened = self._listened_set(user_id_str)
        rec_ids, rec_scores = [int(i) for i in indices[0] if i not in listened], [float(s) for i, s in zip(indices[0], scores[0]) if i not in listened]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n, lambda_=0.4)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=user_id_str, exclude_history=exclude_history)
        return df

    def recommend_next_in_session(self, session_msids, n=10, decay=0.8, filter_session=True, auto_refresh: bool = False, exclude_history: bool = True):
        found = [(m, self.item2idx[m]) for m in session_msids if m in self.item2idx]
        if not found: return self.popular_items(n)
        n_f = len(found)
        w = np.array([decay**(n_f-1-k) for k in range(n_f)], dtype=np.float32)
        query_vec = self._proxy_vector_from_items(np.array([iid for _,iid in found], dtype=np.int32), weights=w/w.sum())
        scores, indices = self.index.search(query_vec, n*4)
        session_set = {iid for _,iid in found} if filter_session else set()
        rec_ids, rec_scores = [int(i) for i in indices[0] if i not in session_set], [float(s) for i, s in zip(indices[0], scores[0]) if i not in session_set]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n, lambda_=0.7)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=None, exclude_history=exclude_history)
        return df

    def recommend_by_artist(self, artist_name, n=10, expand=True, k_similar_artists=5, auto_refresh: bool = False, exclude_history: bool = True):
        a_lower  = artist_name.strip().lower()
        all_iids = list(self._artist2items.get(a_lower, []))
        if expand:
            a_pos = self._artist_name_to_pos.get(a_lower)
            if a_pos is not None: a_vec = self._artist_vecs[a_pos:a_pos+1]
            elif all_iids: a_vec = self._proxy_vector_from_items(all_iids[:30])
            else:
                _, cpos = self._content_index.search(self._text_to_content_vec(a_lower), 10)
                cand = self._content_iids[cpos[0]].tolist()
                a_vec = self._proxy_vector_from_items(cand) if cand else None
            if a_vec is not None:
                sim_sc, sim_pos = self._artist_index.search(a_vec if a_vec.ndim==2 else a_vec.reshape(1,-1), k_similar_artists+1)
                for pos, sc in zip(sim_pos[0], sim_sc[0]):
                    sim_name = self._artist_names[int(pos)]
                    if sim_name != a_lower: all_iids.extend(self._artist2items.get(sim_name, [])[:8])
        if not all_iids: return pd.DataFrame()
        item_counts = np.asarray(self.train_matrix.sum(axis=0)).flatten()
        top_iids = sorted(list(dict.fromkeys(all_iids)), key=lambda i: item_counts[i], reverse=True)[:n]
        df = pd.DataFrame([self._to_row(self.idx2item[i], float(item_counts[i])) for i in top_iids])
        df['artist_name'] = [self._iid_to_artist.get(int(i), '') for i in top_iids]
        df.index = range(1, len(df)+1)
        if auto_refresh:
            return self._apply_refresh_logic(df, n=n, user_id_str=None, exclude_history=exclude_history)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # D. EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate_metrics(self, K=20, eval_batch=512, content_alpha=0.0):
        print(f'\n[Evaluate] Đang tính NDCG/Recall@{K} | content_alpha={content_alpha}')
        test_users = np.unique(self.test_matrix.nonzero()[0])
        n_users = len(test_users)
        total_recall = total_precision = total_ndcg = 0.0
        fetch_K = K * 4 if content_alpha > 0 else K * 2
        
        for start in tqdm(range(0, n_users, eval_batch), desc='Evaluating', unit='batch'):
            batch_u = test_users[start:start+eval_batch]
            lgcn_scores_batch, indices_batch = self.index.search(self.user_vectors[batch_u], fetch_K)
            
            for i, uid in enumerate(batch_u):
                actual = set(self.test_matrix[uid].indices)
                if not actual: continue
                train_seen = set(self.train_matrix[uid].indices)
                
                valid_mask = np.isin(indices_batch[i], list(train_seen), invert=True)
                valid_cands, valid_scores = indices_batch[i][valid_mask], lgcn_scores_batch[i][valid_mask]
                
                if content_alpha > 0.0:
                    content_vec = self._get_user_content_vec(self.idx2user[uid])
                    if content_vec is not None:
                        hybrid_scores = []
                        for iid, lgcn_sc in zip(valid_cands, valid_scores):
                            pos = self._msid_to_cold_pos.get(self.idx2item[iid])
                            hybrid_scores.append((1 - content_alpha) * lgcn_sc + content_alpha * float(self._cold_item_vecs[pos] @ content_vec[0]) if pos is not None else lgcn_sc)
                        valid_cands = valid_cands[np.argsort(hybrid_scores)[::-1]]
                
                preds = valid_cands[:K]
                hits = actual & set(preds)
                total_recall += len(hits) / len(actual)
                total_precision += len(hits) / K
                total_ndcg += sum(1.0/math.log2(r+2) for r, p in enumerate(preds) if p in actual) / sum(1.0/math.log2(r+2) for r in range(min(len(actual), K)))
                
        print(f'  => Recall@{K}: {total_recall/n_users:.4f} | Precision@{K}: {total_precision/n_users:.4f} | NDCG@{K}: {total_ndcg/n_users:.4f}')
        return total_recall/n_users, total_precision/n_users, total_ndcg/n_users


# =============================================================================
# CHẠY LOCAL 8GB RAM — Chuyển index_mappings.pkl → SQLite
# =============================================================================
# === CELL 7: CHAY TREN KAGGLE (1 lan) de tao file tai ve local ===
# Sau khi chay: tai /kaggle/working/mappings.db va mappings_small.pkl
import sqlite3

def convert_pkl_to_sqlite(
    model_dir = '/kaggle/input/datasets/b22dckh068donngkhoa/lightgcn-model',
    output_db = '/kaggle/working/mappings.db',
    out_small = '/kaggle/working/mappings_small.pkl',
):
    """
    Tach index_mappings.pkl thanh 2 file:
      mappings_small.pkl : user2idx, item2idx, idx2user, idx2item, global_max_ts (~30MB)
      mappings.db        : item_meta, user_raw_items, user_item_ts (SQLite, lazy-load)
    RAM savings tren local 8GB: 2-3GB -> <30MB RAM cho mappings
    """
    print('[Convert] Loading index_mappings.pkl...')
    m = joblib.load(os.path.join(model_dir, 'index_mappings.pkl'))

    # 1. Cac dict nho -> pkl compact (can RAM de lookup O(1))
    small = {
        'user2idx'     : m['user2idx'],
        'item2idx'     : m['item2idx'],
        'idx2user'     : m['idx2user'],
        'idx2item'     : m['idx2item'],
        'global_max_ts': m.get('global_max_ts', 0.0),
        'config'       : m.get('config', {}),
    }
    joblib.dump(small, out_small, compress=3)
    del small; gc.collect()
    print(f'  OK mappings_small.pkl: {os.path.getsize(out_small)/1e6:.1f} MB')

    # 2. Ba dict lon -> SQLite
    if os.path.exists(output_db): os.remove(output_db)
    con = sqlite3.connect(output_db)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=NORMAL')
    con.execute('PRAGMA cache_size=-65536')
    con.execute('PRAGMA temp_store=MEMORY')

    # Table 1: item_meta
    print('[Convert] Ghi item_meta...')
    con.execute('''CREATE TABLE item_meta (
        msid        TEXT PRIMARY KEY,
        track_name  TEXT,
        artist_name TEXT
    )''')
    batch = [(msid, meta.get('track_name',''), meta.get('artist_name',''))
             for msid, meta in m['item_meta'].items()]
    con.executemany('INSERT OR IGNORE INTO item_meta VALUES (?,?,?)', batch)
    con.commit()
    print(f'  OK item_meta: {len(batch):,} rows')
    del batch; gc.collect()

    # Table 2: user_raw_items
    print('[Convert] Ghi user_raw_items...')
    con.execute('''CREATE TABLE user_raw_items (user_id TEXT, msid TEXT)''')
    batch = []
    for uid_str, msid_set in tqdm(m['user_raw_items'].items(), desc='user_raw_items'):
        for msid in msid_set:
            batch.append((uid_str, msid))
        if len(batch) >= 200_000:
            con.executemany('INSERT INTO user_raw_items VALUES (?,?)', batch)
            con.commit(); batch.clear()
    if batch:
        con.executemany('INSERT INTO user_raw_items VALUES (?,?)', batch)
        con.commit()
    con.execute('CREATE INDEX idx_uri ON user_raw_items(user_id)')
    con.commit()
    n_uri = con.execute('SELECT COUNT(*) FROM user_raw_items').fetchone()[0]
    print(f'  OK user_raw_items: {n_uri:,} rows')
    del batch; gc.collect()

    # Table 3: user_item_ts
    print('[Convert] Ghi user_item_ts_matrix...')
    con.execute('''CREATE TABLE user_item_ts (uid INTEGER, iid INTEGER, ts REAL)''')
    batch = []
    ts_map = m.get('user_item_ts_matrix', {})
    for uid, iid_ts in tqdm(ts_map.items(), desc='user_item_ts'):
        for iid, ts in iid_ts.items():
            batch.append((int(uid), int(iid), float(ts)))
        if len(batch) >= 200_000:
            con.executemany('INSERT INTO user_item_ts VALUES (?,?,?)', batch)
            con.commit(); batch.clear()
    if batch:
        con.executemany('INSERT INTO user_item_ts VALUES (?,?,?)', batch)
        con.commit()
    # Index tren uid (query per user) va iid (GROUP BY cho trending)
    con.execute('CREATE INDEX idx_uit_uid ON user_item_ts(uid)')
    con.execute('CREATE INDEX idx_uit_iid ON user_item_ts(iid)')
    con.commit()
    n_ts = con.execute('SELECT COUNT(*) FROM user_item_ts').fetchone()[0]
    print(f'  OK user_item_ts: {n_ts:,} rows')
    del batch, ts_map, m; gc.collect()

    con.close()
    db_mb = os.path.getsize(output_db)/1e6
    print(f'\n=== Hoan tat! Tai 2 file nay ve local ===')
    print(f'   mappings.db        : {db_mb:.0f} MB')
    print(f'   mappings_small.pkl : {os.path.getsize(out_small)/1e6:.1f} MB')

#convert_pkl_to_sqlite()


# =============================================================================
# LOCAL RECOMMENDER — SQLite Proxy (không sửa class cũ)
# =============================================================================
# === CELL 9: CHAY TREN LOCAL sau khi tai mappings.db + mappings_small.pkl ===

# ---------------------------------------------------------------------------
# A. SQLITE PROXY CLASSES
# ---------------------------------------------------------------------------

class _SqliteItemMeta:
    """Proxy cho self.item_meta {msid -> {track_name, artist_name}}.
    Fetch 1 row khi can, khong nap 813K records vao RAM.
    """
    def __init__(self, con): self._con = con

    def get(self, msid, default=None):
        if default is None: default = {}
        row = self._con.execute(
            'SELECT track_name, artist_name FROM item_meta WHERE msid=?', (msid,)
        ).fetchone()
        return {'track_name': row[0] or '', 'artist_name': row[1] or ''} if row else default

    def __getitem__(self, msid):
        r = self.get(msid)
        if r: return r
        raise KeyError(msid)

    def __contains__(self, msid):
        return self._con.execute(
            'SELECT 1 FROM item_meta WHERE msid=?', (msid,)
        ).fetchone() is not None

    def items(self):
        """Generator stream toan bo -- dung khi build_content_index / build_artist_index."""
        cur = self._con.execute('SELECT msid, track_name, artist_name FROM item_meta')
        while True:
            rows = cur.fetchmany(10_000)
            if not rows: break
            for msid, track, artist in rows:
                yield msid, {'track_name': track or '', 'artist_name': artist or ''}

    def find_by_track(self, track_lower):
        """SQL lookup thay scan O(813K) -- dung cho generate_playlist."""
        rows = self._con.execute(
            'SELECT msid FROM item_meta WHERE LOWER(track_name)=?', (track_lower,)
        ).fetchall()
        return [r[0] for r in rows]


class _SqliteUserRawItems:
    """Proxy cho self.user_raw_items {uid_str -> set of msid}.
    Query theo user khi can, tiet kiem ~300-500MB RAM.
    """
    def __init__(self, con): self._con = con

    def get(self, user_id_str, default=None):
        if default is None: default = set()
        rows = self._con.execute(
            'SELECT msid FROM user_raw_items WHERE user_id=?', (str(user_id_str),)
        ).fetchall()
        return {r[0] for r in rows} if rows else default

    def __getitem__(self, user_id_str):
        return self.get(str(user_id_str), set())

    def __contains__(self, user_id_str):
        return self._con.execute(
            'SELECT 1 FROM user_raw_items WHERE user_id=?', (str(user_id_str),)
        ).fetchone() is not None

    def values(self):
        """Generator stream theo user -- dung khi build_content_index lay valid_msids."""
        cur = self._con.execute(
            'SELECT user_id, msid FROM user_raw_items ORDER BY user_id'
        )
        current_uid, current_set = None, set()
        while True:
            rows = cur.fetchmany(50_000)
            if not rows:
                if current_uid is not None: yield current_set
                break
            for uid, msid in rows:
                if uid != current_uid:
                    if current_uid is not None: yield current_set
                    current_uid, current_set = uid, set()
                current_set.add(msid)


class _SqliteUserItemTs:
    """Proxy cho self.user_item_ts_matrix {uid_int -> {iid_int -> ts}}.
    Query theo uid khi can, tiet kiem ~500MB-1GB RAM.
    """
    def __init__(self, con): self._con = con

    def get(self, uid, default=None):
        if default is None: default = {}
        rows = self._con.execute(
            'SELECT iid, ts FROM user_item_ts WHERE uid=?', (int(uid),)
        ).fetchall()
        return {r[0]: r[1] for r in rows} if rows else default

    def __getitem__(self, uid):
        return self.get(int(uid), {})

    def __contains__(self, uid):
        return self._con.execute(
            'SELECT 1 FROM user_item_ts WHERE uid=?', (int(uid),)
        ).fetchone() is not None

    def values(self):
        """Generator stream theo uid -- backup cho build_trending khi khong co SQL."""
        cur = self._con.execute('SELECT uid, iid, ts FROM user_item_ts ORDER BY uid')
        current_uid, current_dict = None, {}
        while True:
            rows = cur.fetchmany(100_000)
            if not rows:
                if current_uid is not None: yield current_dict
                break
            for uid, iid, ts in rows:
                if uid != current_uid:
                    if current_uid is not None: yield current_dict
                    current_uid, current_dict = uid, {}
                current_dict[iid] = ts

    def get_filtered(self, uid, start_ts, end_ts):
        """Query co bo loc timestamp -- dung cho recommend_by_timeframe."""
        rows = self._con.execute(
            'SELECT iid, ts FROM user_item_ts WHERE uid=? AND ts>=? AND ts<=?',
            (int(uid), start_ts, end_ts)
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def get_avg_ts_per_item(self, n_items):
        """SQL GROUP BY -- dung cho _build_trending_scores thay Python loop O(95K)."""
        rows = self._con.execute(
            'SELECT iid, AVG(ts) FROM user_item_ts WHERE iid < ? GROUP BY iid',
            (n_items,)
        ).fetchall()
        return {int(r[0]): float(r[1]) for r in rows}


# ---------------------------------------------------------------------------
# B. LOCAL RECOMMENDER -- Subclass, chi override __init__ + 5 method
# ---------------------------------------------------------------------------

class LocalRecommender(AdvancedHybridRecommender):
    """
    Phien ban tiet kiem RAM cho may local 8GB.
    Thay 3 dict lon bang SQLite Proxy -> tiet kiem 1.5-2.5GB RAM.
    Cac method goi y chay nguyen vi proxy implement cung interface dict.
    """

    def __init__(self, model_dir, db_path, small_pkl,
                 cache_dir='./cache/', cold_threshold=5):
        # KHONG goi super().__init__() -- override hoan toan de kiem soat RAM
        t0 = time.time()
        print('[LocalRecommender] Khoi tao cho local 8GB...')
        self.model_dir      = model_dir
        self.cache_dir      = cache_dir
        self.cold_threshold = cold_threshold
        #os.makedirs(cache_dir, exist_ok=True)

        # 1. Embeddings -- mmap_mode='r': anh xa file, khong doc vao RAM
        self.user_vectors = np.load(
            os.path.join(model_dir, 'user_vectors.npy'), mmap_mode='r'
        )
        self.item_vectors = np.load(
            os.path.join(model_dir, 'item_vectors.npy'), mmap_mode='r'
        )
        self.dim = self.item_vectors.shape[1]

        # 2. Dict nho -- van load vao RAM (can O(1) thuong xuyen)
        small = joblib.load(small_pkl)
        self.user2idx      = small['user2idx']   # ~10MB
        self.item2idx      = small['item2idx']   # ~15MB
        self.idx2user      = small['idx2user']   # ~10MB
        self.idx2item      = small['idx2item']   # ~15MB
        self.global_max_ts = small.get('global_max_ts', 0.0)
        del small; gc.collect()
        print(f'  OK small pkl: {len(self.user2idx):,} users | {len(self.item2idx):,} items')

        # 3. SQLite connection

        self._db_path = db_path
        
        
        self._db = sqlite3.connect(
            db_path, 
            check_same_thread=False, 
            isolation_level=None  # <--- BÙA HỘ MỆNH: Bật Autocommit, đọc xong nhả lock ngay!
        )
        
        self._db.execute('PRAGMA cache_size=-32768')  # 32MB page cache
        
        self._db.execute('PRAGMA journal_mode=WAL')
        
        # Dòng này giữ nguyên cũng được, dù mode=ro đã lo liệu việc này
        self._db.execute('PRAGMA query_only=ON')
        print(f'  OK SQLite: {db_path} ({os.path.getsize(db_path)/1e6:.0f} MB)')
        
        # 4. Gan 3 PROXY thay the dict that -- code goc chay khong doi
        self.item_meta           = _SqliteItemMeta(self._db)
        self.user_raw_items      = _SqliteUserRawItems(self._db)
        self.user_item_ts_matrix = _SqliteUserItemTs(self._db)

        # 5. Sparse matrices
        self.train_matrix = sp.load_npz(os.path.join(model_dir, 'train_user_item.npz'))
        self.test_matrix  = sp.load_npz(os.path.join(model_dir, 'test_user_item.npz'))

        # 6. FAISS index chinh -- can copy tu mmap vi FAISS can contiguous
        print('  [FAISS] Building item index...')
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(self.item_vectors))
        self._user_index = None
        print(f'  OK embeddings: user={self.user_vectors.shape} | item={self.item_vectors.shape}')

        # 7. Content & auxiliary indexes (dung master cache neu co)
        self._load_or_build_master_cache()
        print(f'LocalRecommender san sang ({time.time()-t0:.1f}s)')
        
        # 8. Khởi tạo bộ nhớ Refresh
        self._init_refresh_history()

    # Override 1: _build_trending_scores -- SQL GROUP BY thay Python loop ──
    def _build_trending_scores(self, halflife_days=30):
        """SQL: SELECT iid, AVG(ts) GROUP BY iid thay vi Python loop 95K users."""
        t0  = time.time()
        n   = self.item_vectors.shape[0]
        pop = np.asarray(self.train_matrix.sum(axis=0)).flatten().astype(np.float64)

        avg_ts_per_iid = self.user_item_ts_matrix.get_avg_ts_per_item(n)
        avg_ts_arr = np.full(n, float(self.global_max_ts), dtype=np.float64)
        for iid, avg_ts in avg_ts_per_iid.items():
            if 0 <= iid < n: avg_ts_arr[iid] = avg_ts

        hl_sec   = halflife_days * 86400.0
        days_ago = np.clip((self.global_max_ts - avg_ts_arr) / (hl_sec + 1e-9), 0, None)
        trending = pop * np.exp(-days_ago)
        t_max    = trending.max()
        self._trending_scores = (trending / t_max).astype(np.float32) if t_max > 0 else trending.astype(np.float32)
        print(f'  [Trending] SQL GROUP BY | max={self._trending_scores.max():.4f} | {time.time()-t0:.1f}s')

    # Override 2: recommend_by_timeframe -- filter trong SQL ──────────────
    def recommend_by_timeframe(self, user_id_str, start_date, end_date, n=10):
        """Push bo loc timestamp xuong SQL thay vi load toan bo ts_dict vao RAM."""
        uid = self.user2idx.get(str(user_id_str))
        if uid is None: return self.popular_items(n)
        start_ts = pd.Timestamp(start_date).timestamp()
        end_ts   = pd.Timestamp(end_date).timestamp()
        period_rows = self.user_item_ts_matrix.get_filtered(uid, start_ts, end_ts)
        if not period_rows: return pd.DataFrame()
        period_iids = [r[0] for r in period_rows]
        ts_arr = np.array([r[1] for r in period_rows])
        w = np.exp(-(ts_arr.max() - ts_arr) / (30 * 86400))
        period_vec = self._normalize(
            (np.ascontiguousarray(self.item_vectors[period_iids]) * (w/w.sum())[:, None]).sum(axis=0)
        )
        scores, indices = self.index.search(period_vec, n*3)
        period_set = set(period_iids)
        rec_ids, rec_scores = [], []
        for i, s in zip(indices[0], scores[0]):
            if int(i) not in period_set:
                rec_ids.append(int(i)); rec_scores.append(float(s))
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        return df

    # Override 3: generate_playlist -- SQL track lookup thay scan O(813K) ─
    def generate_playlist(self, user_id_str, seed_track_names=None,
                          playlist_name='Goi y danh rieng cho ban', n_songs=15):
        print(f'\n[playlist] "{playlist_name}"')
        uid = self.user2idx.get(str(user_id_str))
        if uid is None: return self.popular_items(n_songs)
        base_vec = np.array(self.user_vectors[uid])
        if seed_track_names:
            seed_vecs = []
            for track in seed_track_names:
                for msid in self.item_meta.find_by_track(track.strip().lower()):
                    if msid in self.item2idx:
                        seed_vecs.append(np.ascontiguousarray(self.item_vectors[self.item2idx[msid]]))
                        break
            if seed_vecs:
                seed_mean = np.mean(seed_vecs, axis=0)
                base_vec  = 0.7 * base_vec + 0.3 * seed_mean / (np.linalg.norm(seed_mean) + 1e-8)
                print(f'  -> Blend 70% user + 30% seed ({len(seed_vecs)} tracks)')
        query_vec = self._normalize(base_vec)
        scores, indices = self.index.search(query_vec, n_songs*4)
        listened  = self._listened_set(user_id_str)
        rec_ids   = [int(i) for i in indices[0] if i not in listened]
        rec_scores = [float(s) for i, s in zip(indices[0], scores[0]) if int(i) not in listened]
        final_ids, final_scores = self._mmr_rerank(rec_ids, rec_scores, n=n_songs, lambda_=0.5)
        df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        df.index = range(1, len(df)+1)
        print(df.columns.tolist())
        print(df.head(5))
        return df

    # Override 4: recommend_similar_to_new_item -- SQL track lookup ────────
    def recommend_similar_to_new_item(self, track_name, artist_name, n=10, include_cold_items=True):
        query = f'{artist_name} {artist_name} {track_name}'
        cvec  = self._text_to_content_vec(query)
        query_lower = f'{track_name} {artist_name}'.lower()
        if include_cold_items:
            n_search = min(n*5, self._cold_item_index.ntotal)
            _, cpos  = self._cold_item_index.search(cvec, n_search)
            results = []
            for pos in cpos[0]:
                msid = self._cold_item_msids[pos]
                meta = self.item_meta.get(msid, {})
                name = (meta.get('track_name','') + ' ' + meta.get('artist_name','')).strip().lower()
                if name == query_lower: continue
                results.append(self._to_row(msid, float(self._cold_item_vecs[pos] @ cvec[0]),
                                            in_model=(msid in self.item2idx)))
                if len(results) >= n: break
            df = pd.DataFrame(results)
        else:
            n_search = min(30, len(self._content_iids))
            _, cpos  = self._content_index.search(cvec, n_search)
            cand_iids = self._content_iids[cpos[0]]
            if not len(cand_iids): return pd.DataFrame()
            proxy_vec = self._proxy_vector_from_items(cand_iids[:15])
            scores, indices = self.index.search(proxy_vec, n*3)
            final_ids, final_scores = [], []
            for iid, sc in zip(indices[0], scores[0]):
                msid = self.idx2item[iid]
                meta = self.item_meta.get(msid, {})
                name = (meta.get('track_name','') + ' ' + meta.get('artist_name','')).strip().lower()
                if name == query_lower: continue
                final_ids.append(int(iid)); final_scores.append(float(sc))
                if len(final_ids) >= n: break
            df = pd.DataFrame([self._to_row(self.idx2item[i], s) for i, s in zip(final_ids, final_scores)])
        if not df.empty:
            df.insert(0, 'similar_to', f'{track_name} ({artist_name})')
            df.index = range(1, len(df)+1)
        return df

    # Override 5: _proxy_vector_from_items -- ascontiguousarray cho mmap ──
    def _proxy_vector_from_items(self, iids, weights=None):
        iids = np.asarray(iids, dtype=np.int32)
        vecs = np.ascontiguousarray(self.item_vectors[iids], dtype=np.float32)
        if weights is not None:
            w = np.asarray(weights, np.float32); w /= w.sum() + 1e-12
            vec = (vecs * w[:, None]).sum(axis=0, keepdims=True)
        else:
            vec = vecs.mean(axis=0, keepdims=True)
        return self._normalize(vec)

    def close(self):
        if hasattr(self, '_db') and self._db:
            self._db.close()
            print('[LocalRecommender] SQLite connection da dong.')
    
    # Hàm bổ sung để tìm kiếm chính xác theo từ khóa trong SQLite
    def search_metadata(self, query_text, n=10):
        search_term = f"%{query_text.lower()}%"
        sql = """
            SELECT msid, track_name, artist_name 
            FROM item_meta 
            WHERE LOWER(track_name) LIKE ? OR LOWER(artist_name) LIKE ? 
            LIMIT ?
        """
        rows = self._db.execute(sql, (search_term, search_term, n * 10)).fetchall()
        
        results = []
        for r in rows:
            results.append({
                'track_name': r[1],  # Đảm bảo r[1] đúng là track_name trong DB
                'artist_name': r[2], # Đảm bảo r[2] đúng là artist_name
                'score': 1.0,
                'in_model': r[0] in self.item2idx,
                'source': 'Metadata Search'
            })
        
        # Tạo DataFrame từ danh sách dict
        df = pd.DataFrame(results)

        # CHỐT CHẶN: Nếu không có kết quả, trả về ngay lập tức
        if df.empty:
            return pd.DataFrame(columns=['track_name', 'artist_name', 'score', 'in_model', 'source'])
        
        # CHỈ CHẠY CÁC DÒNG NÀY NẾU DF KHÔNG RỖNG
        # Normalize trước khi lọc trùng
        df['_name_key'] = df['track_name'].str.strip().str.lower()
        df['_artist_key'] = df['artist_name'].str.strip().str.lower()
        
        df = df.drop_duplicates(subset=["_name_key", "_artist_key"])
        
        # Xóa cột tạm
        df = df.drop(columns=["_name_key", "_artist_key"])
        return df.head(n).reset_index(drop=True)
    # Thêm hàm này vào bên trong class LocalRecommender trong file main.py
    def search_smart(self, query_text, n=10):
        query_text = query_text.lower().strip()
        # 1. Tìm chính xác trong Metadata (SQL)
        search_term = f"%{query_text}%"
        sql = """
            SELECT msid, track_name, artist_name 
            FROM item_meta 
            WHERE LOWER(track_name) LIKE ? OR LOWER(artist_name) LIKE ? 
            LIMIT 50
        """
        rows = self._db.execute(sql, (search_term, search_term)).fetchall()
        
        if rows:
            results = []
            for r in rows:
                results.append({
                    'track_name': r[1], 'artist_name': r[2],
                    'score': 1.0, 'in_model': r[0] in self.item2idx, 'source': 'Metadata'
                })
            df = pd.DataFrame(results).sort_values(by='in_model', ascending=False)
            return df.head(n)
        
        # 2. Nếu không thấy, dùng Vector Search (TF-IDF)
        return self.recommend_cold_content(text_query=query_text, n=n)
    
    def get_user_history(self, user_id_str, limit=30):
        """
        Truy vấn lịch sử nghe nhạc thực tế và hiển thị thông tin bài hát.
        Đã có cơ chế bảo vệ chống IndexError.
        """
        import pandas as pd
        from datetime import datetime
        
        # Lấy ID số của user trong model
        uid = self.user2idx.get(str(user_id_str))
        if uid is None:
            return pd.DataFrame()

        # Truy vấn danh sách iid và timestamp từ SQLite
        sql = "SELECT iid, ts FROM user_item_ts WHERE uid = ? ORDER BY ts DESC LIMIT ?"
        rows = self._db.execute(sql, (uid, limit)).fetchall()

        history = []
        for iid, ts in rows:
            # KIỂM TRA BẢO VỆ: iid phải nằm trong dải chỉ số của danh sách bài hát (idx2item)
            if 0 <= iid < len(self.idx2item):
                msid = self.idx2item[iid]
                meta = self.item_meta.get(msid)
                
                history.append({
                    "Bài hát": meta.get('track_name', 'Không rõ tên'),
                    "Nghệ sĩ": meta.get('artist_name', 'Không rõ nghệ sĩ'),
                    "Ngày nghe": datetime.fromtimestamp(ts).strftime('%d/%m/%Y'),
                    "Giờ nghe": datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                })
        
        # Trả về dưới dạng DataFrame để Streamlit hiển thị bảng đẹp mắt
        return pd.DataFrame(history)
    
    # ─────────────────────────────────────────────────────────────────────────
    # REFRESH HISTORY — Bộ nhớ tránh lặp bài
    # ─────────────────────────────────────────────────────────────────────────

    def _init_refresh_history(self):
        """Khởi tạo dict lưu bài đã gợi ý cho từng user trong phiên."""
        self._refresh_history = defaultdict(set)

    def clear_refresh_history(self, user_id_str):
        """Xóa bộ nhớ refresh của một user cụ thể."""
        self._refresh_history.pop(str(user_id_str), None)

    def _apply_refresh_logic(self, df_pool, n=5, user_id_str=None,
                              exclude_history=True, use_weighted=True, temperature=0.8):
        """
        Bốc thăm ngẫu nhiên theo xác suất điểm số từ df_pool.
        - exclude_history=True : gạch bỏ bài đã xuất hiện trước đó.
        - use_weighted=True    : sampling theo điểm, temperature điều chỉnh độ phân tán.
        Ghi nhớ lại các bài vừa bốc vào _refresh_history.
        """
        if df_pool.empty:
            return df_pool

        df = df_pool.copy()

        # Loại bỏ bài đã hiện
        if exclude_history and user_id_str is not None:
            seen = self._refresh_history.get(str(user_id_str), set())
            if seen:
                df = df[~df['track_name'].isin(seen)]
            # Nếu lọc xong hết sạch → reset và dùng lại pool gốc
            if df.empty:
                self._refresh_history[str(user_id_str)] = set()
                df = df_pool.copy()

        # Bốc thăm
        k = min(n, len(df))
        if use_weighted and len(df) > k:
            scores = df['score'].values.astype(np.float64)
            scores = scores - scores.min() + 1e-8
            scores = np.power(scores, 1.0 / max(temperature, 1e-3))
            probs  = scores / scores.sum()
            chosen = np.random.choice(len(df), size=k, replace=False, p=probs)
            result = df.iloc[sorted(chosen)]
        else:
            result = df.head(k)

        # Ghi nhớ bài vừa bốc
        if user_id_str is not None:
            self._refresh_history[str(user_id_str)].update(result['track_name'].tolist())

        result = result.reset_index(drop=True)
        result.index = range(1, len(result) + 1)
        return result
    
print('LocalRecommender da san sang de khoi tao.')

# =============================================================================
# CHẠY TEST DEMO
# =============================================================================
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Đường dẫn local – tất cả file đều nằm trong thư mục hiện tại
    LOCAL_MODEL_DIR = r'.'                       # thư mục hiện tại
    LOCAL_DB        = r'./mappings.db'           # tên file database
    LOCAL_SMALL_PKL = r'./mappings_small.pkl'    # tên file pickle
    LOCAL_CACHE_DIR = r'.' # thư mục cache (sẽ tự tạo nếu uncomment dòng os.makedirs)

    try:
        rec_local = LocalRecommender(
            model_dir = LOCAL_MODEL_DIR,
            db_path   = LOCAL_DB,
            small_pkl = LOCAL_SMALL_PKL,
            cache_dir = LOCAL_CACHE_DIR,
        )
    except FileNotFoundError as e:
        print(f'LOI: {e}')
        rec_local = None
    except Exception as e:
        print(f'LOI: {e}')
        rec_local = None

    # Demo nhanh
    if rec_local is not None:
        test_uid_idx = np.unique(rec_local.test_matrix.nonzero()[0])[0]
        test_user    = rec_local.idx2user[test_uid_idx]
        uid_ts       = rec_local.user_item_ts_matrix.get(test_uid_idx, {})

        print('\n' + '='*70)
        print(f' DEMO TỔNG HỢP USER: {test_user} (Tier: {rec_local._get_user_tier(test_user)[0]})'.center(70))
        print('='*70)

        # --- Phần 0: Đánh giá độ đo (Metrics) ---
    #    print('\n[0] ĐÁNH GIÁ ĐỘ ĐO (METRICS) K=20:')
    #    print('>> Base LightGCN (alpha=0.0):')
    #    rec_local.evaluate_metrics(K=20, content_alpha=0.0)
    #    print('\n>> Hybrid TF-IDF (alpha=0.25):')
    #    rec_local.evaluate_metrics(K=20, content_alpha=0.25)

        # --- Phần 1: Gợi ý Hybrid và Nội dung ---
        print('\n[1] recommend_hybrid (LightGCN + TF-IDF) - content_alpha=0.25:')
        display(rec_local.recommend_hybrid(test_user, n=5, artist_limit=2))

        print('\n[2] recommend_inclusive (Top Hit + Hidden Gem):')
        display(rec_local.recommend_inclusive(test_user, n_warm=3, n_cold=2))

        print('\n[3] recommend_cold_content (Thuần TF-IDF):')
        display(rec_local.recommend_cold_content(user_id_str=test_user, n=5))

        print('\n[4] recommend_similar_to_new_item (Khi thêm bài mới):')
        display(rec_local.recommend_similar_to_new_item('Creep', 'Radiohead', n=5))

        # --- Phần 2: Các tính năng nâng cao (Khôi phục từ gốc) ---
        print('\n[5] generate_playlist (Pha trộn gu User và bài Seed "Lightbulb Sun"):')
        display(rec_local.generate_playlist(test_user, seed_track_names=['Lightbulb Sun'], n_songs=5))

        print('\n[6] recommend_realtime (User vừa nghe 3 bài liên tiếp):')
        recent_msids = list(rec_local.item2idx.keys())[:3]
        display(rec_local.recommend_realtime(test_user, recent_listened_msids=recent_msids, n=5, alpha=0.5))

        print('\n[7] recommend_trending (Xu hướng chung toàn hệ thống):')
        display(rec_local.recommend_trending(user_id_str=None, n=5, personal_weight=0.0))

        print('\n[8] recommend_discovery (Gợi ý thoát khỏi vùng an toàn - Khám phá mới):')
        display(rec_local.recommend_discovery(test_user, n=5, serendipity=0.4))

        print('\n[9] recommend_similar_users (Gợi ý từ cộng đồng người dùng tương tự):')
        display(rec_local.recommend_similar_users(test_user, n=5, k_users=20))

        if uid_ts:
            print('\n[10] recommend_next_in_session (Dự đoán bài hát tiếp theo của Session):')
            sorted_by_ts  = sorted(uid_ts.items(), key=lambda x: x[1], reverse=True)
            session_msids = [rec_local.idx2item[i] for i, _ in sorted_by_ts[:5] if i < len(rec_local.idx2item)]
            display(rec_local.recommend_next_in_session(session_msids, n=5, decay=0.8))

            if rec_local.global_max_ts > 0:
                print('\n[11] recommend_by_timeframe (Lọc sở thích nghe trong 6 tháng qua):')
                end_dt = pd.Timestamp(rec_local.global_max_ts, unit='s')
                start_dt = end_dt - pd.Timedelta(days=180)
                display(rec_local.recommend_by_timeframe(test_user, str(start_dt.date()), str(end_dt.date()), n=5))

        # dong connection khi xong -- bo comment neu can
        # rec_local.close()
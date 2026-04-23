"""
04_lgcn_engine.py — LAYER 2C: "Nhà Máy C"
===========================================
Đầu vào : data/outputs/clean_data/ (từ Layer 1)
Đầu ra  :
  - data/models/user_vectors.npy
  - data/models/item_vectors.npy
  - data/models/index_mappings.pkl
  - data/models/train_matrix.npz

Thuật toán:
  - InteractionAccumulator (recency-weighted, global max timestamp)
  - Temporal Train/Test split (80/20 theo timestamp)
  - LightGCN (3 layers, BPR loss, popularity-based negative sampling)
  - FAISS index cho fast ANN search
  
Chạy: python 04_lgcn_engine.py
"""
import os, gc, time, math, pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from torch.utils.data import Dataset, DataLoader
import faiss

import sys
sys.path.insert(0, "/Workspace/Users/truongtd.b22kh130@stu.ptit.edu.vn/DataMining-/modules/unified_music_system")
from config import OUTPUT_DIR, MODEL_DIR, LGCN_CONFIG

os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# DATA ACCUMULATOR
# ──────────────────────────────────────────────────────────────────────────────

class InteractionAccumulator:
    """
    Tích lũy tương tác từ nhiều CSV.
    Lưu (count, last_ts_unix) per (user, item) — recency weight đúng với global max.
    """
    def __init__(self):
        self.user2idx = {}; self.item2idx = {}
        self.idx2user = []; self.idx2item = []
        self.item_meta = {}
        self._raw = defaultdict(lambda: [0, 0.0])
        self.global_max_ts = 0.0
        self.user_raw_items = defaultdict(set)
        self.user_item_ts_matrix = None
        self.n_rows_processed = 0
        self._n_ts_missing = 0

    def _get_or_create(self, mapping, reverse, key):
        if key not in mapping:
            mapping[key] = len(reverse)
            reverse.append(key)
        return mapping[key]

    def add_dataframe(self, df: pd.DataFrame):
        required = ["user_id", "recording_msid", "track_name", "artist_name"]
        has_ts   = "timestamp" in df.columns
        df = df[[*required, *( ["timestamp"] if has_ts else [])]].dropna(subset=required).copy()

        # Metadata
        meta = df[["recording_msid", "track_name", "artist_name"]].drop_duplicates("recording_msid")
        for r in meta.itertuples(index=False):
            if r.recording_msid not in self.item_meta:
                self.item_meta[r.recording_msid] = {
                    "track_name":  str(r.track_name).strip(),
                    "artist_name": str(r.artist_name).strip(),
                }

        df["user_id"] = df["user_id"].astype(str)

        if has_ts:
            df["ts_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
            n_before = len(df)
            df = df[df["ts_parsed"].notna()].copy()
            self._n_ts_missing += n_before - len(df)
            df["ts_unix"] = df["ts_parsed"].astype("int64") / 1e9
            chunk_max = df["ts_unix"].max()
            if chunk_max > self.global_max_ts:
                self.global_max_ts = chunk_max
        else:
            df["ts_unix"] = 0.0

        agg = df.groupby(["user_id", "recording_msid"], sort=False).agg(
            count=("recording_msid", "count"),
            last_ts=("ts_unix", "max"),
        ).reset_index()

        for r in agg.itertuples(index=False):
            uid_str, msid = r.user_id, r.recording_msid
            self._get_or_create(self.user2idx, self.idx2user, uid_str)
            self._get_or_create(self.item2idx, self.idx2item, msid)
            entry = self._raw[(uid_str, msid)]
            entry[0] += int(r.count)
            if r.last_ts > entry[1]:
                entry[1] = float(r.last_ts)
            self.user_raw_items[uid_str].add(msid)

        self.n_rows_processed += len(df)

    def build_matrix(self, min_interactions: int = 20):
        hl = LGCN_CONFIG["recency_halflife"]
        t0 = time.time()
        rows, cols, data = [], [], []
        for (uid_str, msid), (count, last_ts) in self._raw.items():
            if uid_str not in self.user2idx or msid not in self.item2idx:
                continue
            uid = self.user2idx[uid_str]
            iid = self.item2idx[msid]
            if self.global_max_ts > 0 and last_ts > 0:
                days_ago = (self.global_max_ts - last_ts) / 86400.0
                recency  = math.exp(-days_ago / hl)
            else:
                recency = 1.0
            w = float(count) * recency
            rows.append(uid); cols.append(iid); data.append(w)

        mat = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.idx2user), len(self.idx2item)),
            dtype=np.float32,
        )
        print(f"  Ma trận ban đầu: {mat.shape[0]:,}×{mat.shape[1]:,} | {mat.nnz:,} interactions")

        if min_interactions > 1:
            mat_bin = (mat > 0).astype(np.float32)
            ic_cnt = np.asarray(mat_bin.sum(axis=0)).flatten()
            uc_cnt = np.asarray(mat_bin.sum(axis=1)).flatten()
            vi = ic_cnt >= min_interactions
            vu = uc_cnt >= min_interactions
            mat = mat[vu][:, vi]
            self.idx2user = [u for u, v in zip(self.idx2user, vu) if v]
            self.idx2item = [i for i, v in zip(self.idx2item, vi) if v]
            self.user2idx = {u: i for i, u in enumerate(self.idx2user)}
            self.item2idx = {i: j for j, i in enumerate(self.idx2item)}
            print(f"  Sau lọc: {mat.shape[0]:,} users | {mat.shape[1]:,} items | {mat.nnz:,} interactions")

        self.user_item_ts_matrix = defaultdict(dict)
        for (uid_str, msid), (_, last_ts) in self._raw.items():
            if uid_str in self.user2idx and msid in self.item2idx:
                self.user_item_ts_matrix[self.user2idx[uid_str]][self.item2idx[msid]] = last_ts
        print(f"[build_matrix] Hoàn tất: {time.time()-t0:.1f}s")
        return mat


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL SPLIT
# ──────────────────────────────────────────────────────────────────────────────

def temporal_split(user_item_matrix, accumulator, test_ratio=0.2, min_items=5):
    n_users = user_item_matrix.shape[0]
    indptr  = user_item_matrix.indptr
    idx_arr = user_item_matrix.indices
    data_arr= user_item_matrix.data

    tr_r, tr_c, tr_d = [], [], []
    te_r, te_c, te_d = [], [], []

    for uid in range(n_users):
        s, e   = indptr[uid], indptr[uid + 1]
        items  = idx_arr[s:e]
        vals   = data_arr[s:e]

        if len(items) < min_items:
            for iid, v in zip(items, vals):
                tr_r.append(uid); tr_c.append(int(iid)); tr_d.append(float(v))
            continue

        ts_dict = accumulator.user_item_ts_matrix.get(uid, {})
        ts_vals = [ts_dict.get(int(iid), 0.0) for iid in items]
        order   = sorted(range(len(items)), key=lambda k: ts_vals[k])
        n_test  = max(1, int(len(items) * test_ratio))

        for rank, k in enumerate(order):
            iid, v = int(items[k]), float(vals[k])
            if rank < len(order) - n_test:
                tr_r.append(uid); tr_c.append(iid); tr_d.append(v)
            else:
                te_r.append(uid); te_c.append(iid); te_d.append(v)

    shape   = user_item_matrix.shape
    train_m = sp.csr_matrix((tr_d, (tr_r, tr_c)), shape=shape, dtype=np.float32)
    test_m  = sp.csr_matrix((te_d, (te_r, te_c)), shape=shape, dtype=np.float32)
    print(f"  Train: {train_m.nnz:,} | Test: {test_m.nnz:,}")
    return train_m, test_m


# ──────────────────────────────────────────────────────────────────────────────
# LIGHTGCN MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_graph(user_item_matrix):
    n_users, n_items = user_item_matrix.shape
    n_nodes = n_users + n_items
    rows, cols   = user_item_matrix.nonzero()
    cols_shifted = cols + n_users
    edge_i = np.concatenate([rows, cols_shifted])
    edge_j = np.concatenate([cols_shifted, rows])
    degree      = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(degree, edge_i, 1.0)
    d_inv_sqrt  = np.where(degree > 0, degree ** -0.5, 0.0)
    edge_weight = d_inv_sqrt[edge_i] * d_inv_sqrt[edge_j]
    indices = torch.from_numpy(np.vstack((edge_i, edge_j))).long()
    values  = torch.from_numpy(edge_weight)
    graph   = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()
    return graph.to(DEVICE)


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=128, n_layers=3):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.n_layers = n_layers
        self.emb_user = nn.Embedding(n_users, emb_dim)
        self.emb_item = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.emb_user.weight, std=0.1)
        nn.init.normal_(self.emb_item.weight, std=0.1)

    def forward(self, graph):
        all_emb = torch.cat([self.emb_user.weight, self.emb_item.weight])
        embs    = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        out = torch.mean(torch.stack(embs, dim=1), dim=1)
        return torch.split(out, [self.n_users, self.n_items])


class BPRDataset(Dataset):
    def __init__(self, user_item_matrix, neg_pool_size=2_000_000):
        coo = user_item_matrix.tocoo()
        weights = np.asarray(coo.data, dtype=np.float32)
        self.users  = torch.LongTensor(coo.row)
        self.items  = torch.LongTensor(coo.col)
        self.weights= torch.FloatTensor(weights / weights.max())
        n_items     = user_item_matrix.shape[1]
        # Popularity-based negative pool
        item_pop    = np.asarray(user_item_matrix.sum(axis=0)).flatten().astype(np.float64)
        item_pop    = item_pop ** 0.75
        item_pop   /= item_pop.sum()
        self.neg_pool = np.random.choice(n_items, size=neg_pool_size, p=item_pop)
        self._neg_ptr = 0

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg = int(self.neg_pool[self._neg_ptr % len(self.neg_pool)])
        self._neg_ptr += 1
        return self.users[idx], self.items[idx], torch.LongTensor([neg])[0], self.weights[idx]


def bpr_loss(u, pos, neg, u0, pos0, neg0):
    pos_sc = (u * pos).sum(1)
    neg_sc = (u * neg).sum(1)
    bpr    = -F_torch.logsigmoid(pos_sc - neg_sc).mean()
    reg    = (u0.norm(2).pow(2) + pos0.norm(2).pow(2) + neg0.norm(2).pow(2)) / (2 * len(u))
    return bpr, reg


# ──────────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_lgcn(train_matrix, accumulator):
    cfg   = LGCN_CONFIG
    n_u   = train_matrix.shape[0]
    n_i   = train_matrix.shape[1]
    graph = build_graph(train_matrix)
    model = LightGCN(n_u, n_i, cfg["emb_dim"], cfg["n_layers"]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"], eta_min=1e-5
    )

    dataset = BPRDataset(train_matrix, cfg["neg_pool_size"])
    loader  = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=(DEVICE == "cuda"),
    )

    print(f"\nLightGCN | {n_u:,} users | {n_i:,} items | device={DEVICE}")
    ckpt_dir = os.path.join(MODEL_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False):
            u_ids, p_ids, n_ids, w = [b.to(DEVICE) for b in batch]
            u_emb, i_emb = model(graph)
            u   = u_emb[u_ids];   pos = i_emb[p_ids];   neg = i_emb[n_ids]
            u0  = model.emb_user(u_ids); pos0 = model.emb_item(p_ids); neg0 = model.emb_item(n_ids)
            bpr, reg = bpr_loss(u * w.unsqueeze(1), pos, neg, u0, pos0, neg0)
            loss = bpr + cfg["decay"] * reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            total_loss += loss.item()
        sched.step()
        avg_loss = total_loss / max(len(loader), 1)
        print(f"  Epoch {epoch:02d} | loss={avg_loss:.4f} | lr={sched.get_last_lr()[0]:.2e}")

        if epoch % 5 == 0 or epoch == cfg["epochs"]:
            ckpt_path = os.path.join(ckpt_dir, f"lgcn_epoch{epoch:02d}.pt")
            torch.save(model.state_dict(), ckpt_path)

    return model, graph


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "=" * 60)
    print("LAYER 2C — LIGHTGCN ENGINE")
    print("=" * 60)

    # 1. Đọc dữ liệu sạch từ Tầng Silver ──────────────────────────────
    print("\n[1/4] Đọc dữ liệu từ default.silver_unified_logs...")
    
    # Gọi Spark Session (vì file này chạy độc lập)
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    from pyspark.sql import functions as F
    
    # --- 1. Đọc dữ liệu sạch từ Tầng Silver (Unity Catalog) ---
    print("\n[1/4] Đọc dữ liệu từ bảng music_ai_workspace.default.silver_unified_logs...")
    
    # Sử dụng Spark để gom nhóm trước, giảm hàng chục triệu dòng xuống còn các cặp (user, item) duy nhất
    # Việc này giúp toPandas() nhẹ hơn gấp nhiều lần!
    df_silver_agg = (spark.table("music_ai_workspace.default.silver_unified_logs")
        .groupBy("user_id", "recording_msid")
        .agg(
            F.max("timestamp").alias("timestamp"), # Lấy thời gian tương tác cuối
            F.first("artist_name").alias("artist_name")
        )
        .withColumn("track_name", F.col("recording_msid")) # Vá lỗ hổng track_name
    )

    print("  Đang tải dữ liệu đã tổng hợp về RAM máy chủ...")
    pdf = df_silver_agg.toPandas()

    # --- 2. Nạp vào Accumulator ---
    # Vì dữ liệu đã được Spark gom nhóm (Aggregated), chúng ta không cần chia chunk nữa
    # hoặc nếu muốn giữ logic chunk cho an toàn thì code sẽ như sau:
    chunk_size = LGCN_CONFIG.get("chunk_size", 100000)
    for i in tqdm(range(0, len(pdf), chunk_size), desc="Nạp interactions vào Graph"):
        chunk = pdf.iloc[i : i + chunk_size]
        accumulator.add_dataframe(chunk)
        
    del pdf
    import gc
    gc.collect()

    print(f"  Đọc xong: {accumulator.n_rows_processed:,} rows")
    user_item_matrix = accumulator.build_matrix(LGCN_CONFIG["min_interactions"])

    # 2. Temporal split
    print("\n[2/4] Temporal split...")
    train_matrix, test_matrix = temporal_split(
        user_item_matrix, accumulator, test_ratio=0.2, min_items=5
    )

    # 3. Huấn luyện
    print("\n[3/4] Huấn luyện LightGCN...")
    model, graph = train_lgcn(train_matrix, accumulator)

    # 4. Trích xuất vectors & lưu
    print("\n[4/4] Lưu embeddings & index mappings...")
    model.eval()
    with torch.no_grad():
        u_emb, i_emb = model(graph)
        user_vecs = F_torch.normalize(u_emb, p=2, dim=1).cpu().numpy().astype(np.float32)
        item_vecs = F_torch.normalize(i_emb, p=2, dim=1).cpu().numpy().astype(np.float32)

    np.save(os.path.join(MODEL_DIR, "user_vectors.npy"), user_vecs)
    np.save(os.path.join(MODEL_DIR, "item_vectors.npy"), item_vecs)
    sp.save_npz(os.path.join(MODEL_DIR, "train_matrix.npz"), train_matrix)
    sp.save_npz(os.path.join(MODEL_DIR, "test_matrix.npz"),  test_matrix)

    index_mappings = {
        "user2idx"          : accumulator.user2idx,
        "item2idx"          : accumulator.item2idx,
        "idx2user"          : accumulator.idx2user,
        "idx2item"          : accumulator.idx2item,
        "item_meta"         : accumulator.item_meta,
        "user_item_ts_matrix": dict(accumulator.user_item_ts_matrix),
        "global_max_ts"     : accumulator.global_max_ts,
    }
    with open(os.path.join(MODEL_DIR, "index_mappings.pkl"), "wb") as f:
        pickle.dump(index_mappings, f)

    print(f"  ✅ user_vectors.npy  : {user_vecs.shape}")
    print(f"  ✅ item_vectors.npy  : {item_vecs.shape}")
    print(f"  ✅ index_mappings.pkl")
    print("\n✅ Layer 2C — LightGCN Engine hoàn tất!")


if __name__ == "__main__":
    run()

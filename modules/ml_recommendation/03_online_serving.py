# Databricks notebook source
# MAGIC %md
# MAGIC # Hệ thống Gợi ý (Online Serving Logic)
# MAGIC **Mục tiêu:** Cung cấp các function API cho việc gợi ý (Recommend) sử dụng FAISS kết hợp kỹ thuật MMR để đa dạng hóa, và Late Fusion để trộn vector lịch sử ngắn hạn.

# COMMAND ----------

import os
import joblib
import numpy as np
import pandas as pd
import faiss

ARTIFACTS_DIR = "/dbfs/FileStore/recommender_artifacts"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Khởi tạo Engine Gợi ý (Load Artifacts)

# COMMAND ----------

class LightGCNRecommender:
    def __init__(self, model_dir):
        print(f"Loading mô hình từ {model_dir}...")
        
        self.user_vectors = np.load(os.path.join(model_dir, "user_vectors.npy"))
        self.item_vectors = np.load(os.path.join(model_dir, "item_vectors.npy"))
        
        mappings = joblib.load(os.path.join(model_dir, "index_mappings.pkl"))
        self.user2idx  = mappings['user2idx']
        self.item2idx  = mappings['item2idx']
        self.idx2user  = mappings['idx2user']
        self.idx2item  = mappings['idx2item']
        self.item_meta = mappings['item_meta']
        
        self.index = faiss.read_index(os.path.join(model_dir, "item_faiss.index"))
        
        print(f"Loaded: {len(self.user2idx)} Users | {len(self.item2idx)} Items")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        v = vec.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(v)
        return v
    
    def _mmr_rerank(self, rec_ids, scores, n=10, lambda_=0.6):
        # Đa dạng hóa danh sách bằng Maximal Marginal Relevance
        if len(rec_ids) == 0: return [], []
        
        rec_ids = np.array(rec_ids)
        scores = np.array(scores)
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
            best_local = int(np.argmax(mmr_scores))
            selected.append(unsel[best_local])
            unsel.pop(best_local)
            
        return rec_ids[selected].tolist(), scores[selected].tolist()

# Khởi tạo Global Instance
recommender = LightGCNRecommender(ARTIFACTS_DIR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Các hàm Gợi ý (API Endpoints)

# COMMAND ----------

# Hàm 1: Gợi ý cơ bản (Đã áp dụng MMR)
def get_recommendation(user_id_str, top_n=10):
    uid = recommender.user2idx.get(str(user_id_str))
    if uid is None:
        return "User mới (Cold-Start) -> Trả về danh sách Top Trending (Tính từ Spark SQL)"
        
    user_vec = recommender.user_vectors[uid:uid+1]
    
    # FAISS search lấy gấp 4 lần để filter MMR
    scores, indices = recommender.index.search(user_vec, top_n * 4)
    
    # Áp dụng MMR
    final_ids, final_scores = recommender._mmr_rerank(indices[0], scores[0], n=top_n, lambda_=0.5)
    
    results = []
    for iid, sc in zip(final_ids, final_scores):
        msid = recommender.idx2item[iid]
        meta = recommender.item_meta.get(msid, {})
        results.append({
            "track_name": meta.get("track_name"),
            "artist_name": meta.get("artist_name"),
            "score": round(float(sc), 4)
        })
    return pd.DataFrame(results)

# Test hàm
display(get_recommendation(user_id_str="13"))

# COMMAND ----------

# Hàm 2: Real-time Late Fusion (Kết hợp bài hát vừa nghe)
def get_realtime_recommendation(user_id_str, recent_msids, top_n=10, alpha=0.4):
    uid = recommender.user2idx.get(str(user_id_str))
    
    # Lấy vector trung bình của các bài vừa nghe (Short-term)
    valid_msids = [m for m in recent_msids if m in recommender.item2idx]
    if not valid_msids:
        return get_recommendation(user_id_str, top_n)
        
    short_vec = np.mean([recommender.item_vectors[recommender.item2idx[m]] for m in valid_msids], axis=0)
    
    # Trộn với Long-term vector của user
    if uid is not None:
        long_vec = recommender.user_vectors[uid]
        fused_vec = (1 - alpha) * long_vec + alpha * short_vec
    else:
        fused_vec = short_vec # User mới nhưng có bài vừa nghe
        
    query_vec = recommender._normalize(fused_vec)
    
    scores, indices = recommender.index.search(query_vec, top_n * 4)
    final_ids, final_scores = recommender._mmr_rerank(indices[0], scores[0], n=top_n)
    
    results = []
    for iid, sc in zip(final_ids, final_scores):
        msid = recommender.idx2item[iid]
        meta = recommender.item_meta.get(msid, {})
        results.append({
            "track_name": meta.get("track_name"),
            "artist_name": meta.get("artist_name"),
            "score": round(float(sc), 4)
        })
    return pd.DataFrame(results)

# Test Late Fusion
recent_tracks = list(recommender.item_meta.keys())[:3]
display(get_realtime_recommendation(user_id_str="13", recent_msids=recent_tracks))

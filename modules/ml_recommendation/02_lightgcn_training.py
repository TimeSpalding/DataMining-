# Databricks notebook source
# MAGIC %md
# MAGIC # Huấn luyện mô hình LightGCN và Indexing FAISS
# MAGIC **Mục tiêu:** Xây dựng và huấn luyện mô hình học sâu LightGCN trên ma trận tương tác. Sau đó trích xuất Embeddings và lưu vào FAISS Index để phục vụ truy vấn tốc độ cao.
# MAGIC **Output:** Các mảng Numpy chứa User/Item Vectors và FAISS Index file.

# COMMAND ----------

import os
import time
import joblib
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import mlflow
from torch.utils.data import Dataset, DataLoader

ARTIFACTS_DIR = "/dbfs/FileStore/recommender_artifacts"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Ma trận & Định nghĩa Dataloader

# COMMAND ----------

train_matrix = sp.load_npz(os.path.join(ARTIFACTS_DIR, "train_matrix.npz"))
num_users, num_items = train_matrix.shape

# Dataloader lấy mẫu Positive (tương tác) và Negative (không tương tác)
class BPRDataset(Dataset):
    def __init__(self, matrix):
        self.users, self.pos_items = matrix.nonzero()
        self.num_items = matrix.shape[1]
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.pos_items[idx]
        # Lấy mẫu ngẫu nhiên negative
        j = np.random.randint(self.num_items)
        return u, i, j

dataset = BPRDataset(train_matrix)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Định nghĩa Kiến trúc LightGCN

# COMMAND ----------

# Hàm tạo adjacency matrix chuẩn hóa cho LightGCN
def build_adj_matrix(matrix):
    n_users, n_items = matrix.shape
    adj = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj = adj.tolil()
    R = matrix.tolil()
    
    adj[:n_users, n_users:] = R
    adj[n_users:, :n_users] = R.T
    adj = adj.todok()
    
    rowsum = np.array(adj.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj).dot(d_mat)
    norm_adj = norm_adj.tocsr()
    
    # Đưa vào Sparse Tensor của PyTorch
    coo = norm_adj.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(i, v, coo.shape)

norm_adj_tensor = build_adj_matrix(train_matrix)
if torch.cuda.is_available():
    norm_adj_tensor = norm_adj_tensor.cuda()

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.layers = layers
        
    def forward(self, adj):
        ego_embeddings = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_emb, i_emb = torch.split(all_embeddings, [self.user_emb.num_embeddings, self.item_emb.num_embeddings])
        return u_emb, i_emb

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Huấn luyện Model (MLflow Tracking)

# COMMAND ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightGCN(num_users, num_items, emb_dim=64, layers=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10

mlflow.set_experiment("/Users/truongtrinhdac03@gmail.com/LightGCN_Recommendation")

with mlflow.start_run(run_name="LightGCN_Training"):
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for u, i, j in dataloader:
            u, i, j = u.to(device), i.to(device), j.to(device)
            
            u_emb, i_emb = model(norm_adj_tensor)
            
            user = u_emb[u]
            pos_item = i_emb[i]
            neg_item = i_emb[j]
            
            # BPR Loss
            pos_scores = torch.mul(user, pos_item).sum(dim=1)
            neg_scores = torch.mul(user, neg_item).sum(dim=1)
            bpr_loss = -nn.LogSigmoid()(pos_scores - neg_scores).mean()
            
            optimizer.zero_grad()
            bpr_loss.backward()
            optimizer.step()
            
            total_loss += bpr_loss.item()
            
        avg_loss = total_loss / len(dataloader)
        mlflow.log_metric("bpr_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1}/{EPOCHS} | BPR Loss: {avg_loss:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Trích xuất Embeddings và Đưa vào FAISS

# COMMAND ----------

model.eval()
with torch.no_grad():
    final_u, final_i = model(norm_adj_tensor)
    user_vectors = final_u.cpu().numpy().astype(np.float32)
    item_vectors = final_i.cpu().numpy().astype(np.float32)

# L2-normalize cho Cosine Similarity (IndexFlatIP)
faiss.normalize_L2(user_vectors)
faiss.normalize_L2(item_vectors)

# Lưu vector xuống đĩa
np.save(os.path.join(ARTIFACTS_DIR, "user_vectors.npy"), user_vectors)
np.save(os.path.join(ARTIFACTS_DIR, "item_vectors.npy"), item_vectors)

# Build FAISS Index
dim = item_vectors.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(item_vectors)

# Lưu Index FAISS
faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "item_faiss.index"))

print("Đã lưu Vector Embeddings và FAISS Index thành công!")

import subprocess
import sys

def install_packages():
    packages = ["torch", "faiss-cpu", "scikit-learn", "joblib"]
    print("Đang tự động cài đặt các thư viện:", packages)
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)

install_packages()


import os
import time
import glob
import joblib
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import mlflow
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import shutil

ARTIFACTS_DIR = "/Volumes/workspace/default/recommender_artifacts"

TMP_DIR = "/tmp/recommender_training"
os.makedirs(TMP_DIR, exist_ok=True)


# 1. Load Data và Định nghĩa DataLoader


print(f"Loading data từ {ARTIFACTS_DIR}...")
train_matrix = sp.load_npz(os.path.join(ARTIFACTS_DIR, "train_matrix.npz"))
mappings = joblib.load(os.path.join(ARTIFACTS_DIR, "index_mappings.pkl"))

num_users, num_items = train_matrix.shape
item_meta = mappings['item_meta']
item2idx = mappings['item2idx']

# Dataloader lấy mẫu Positive (tương tác) và Negative (không tương tác)
class WeightedBPRDataset(Dataset):
    def __init__(self, matrix, neg_pool_size=2000000):
        coo = matrix.tocoo()
        self.users = coo.row.astype(np.int32)
        self.pos_items = coo.col.astype(np.int32)
        self.weights = coo.data.astype(np.float32)
        self.num_items = matrix.shape[1]
        
        # Tạo phân phối Negative Sampling dựa trên độ phổ biến (Popularity)
        item_counts = np.asarray(matrix.sum(axis=0)).flatten()
        neg_probs = np.power(item_counts + 1e-8, 0.75)
        neg_probs /= neg_probs.sum()
        
        print(f"Precomputing negative pool ({neg_pool_size} samples)...")
        self._pool = np.random.choice(self.num_items, size=neg_pool_size, p=neg_probs).astype(np.int32)
        self._pool_size = neg_pool_size
        
    def __len__(self):
        return len(self.users)
        
    def __getitem__(self, idx):
        u = int(self.users[idx])
        i = int(self.pos_items[idx])
        j = int(self._pool[np.random.randint(self._pool_size)])
        return u, i, j

dataset = WeightedBPRDataset(train_matrix)

# Weighted Positive Sampling giống bản gốc
sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.from_numpy(dataset.weights),
    num_samples=len(dataset),
    replacement=True
)

# Đưa num_workers=0 để chạy an toàn trên Serverless Container (Tránh lỗi vỡ Shared Memory /dev/shm)
dataloader = DataLoader(dataset, batch_size=200000, sampler=sampler, num_workers=0)


# 2. Kiến trúc LightGCN


# Hàm tạo adjacency matrix chuẩn hóa cho LightGCN
def build_adj_matrix(matrix):
    n_users, n_items = matrix.shape
    
    # Thay vì dùng DOK và LIL tốn RAM, dùng Block Matrix trực tiếp tạo CSR
    adj = sp.bmat([[None, matrix], [matrix.T, None]], format='csr', dtype=np.float32)
    
    # Tính Degree Matrix (D^-0.5)
    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv = np.power(rowsum, -0.5)
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    # Chuẩn hóa: D^-0.5 * A * D^-0.5
    norm_adj = d_mat.dot(adj).dot(d_mat)
    
    # Chuyển đổi sang PyTorch Sparse Tensor
    coo = norm_adj.tocoo()
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))
    v = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(i, v, coo.shape)

print("Building Adjacency Matrix...")
norm_adj_tensor = build_adj_matrix(train_matrix)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    norm_adj_tensor = norm_adj_tensor.cuda()

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=128, layers=3, pretrained_user_emb=None, pretrained_item_emb=None):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        
        # Khởi tạo mặc định
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        
        # Nếu có Pre-trained Weights (Fine-tuning)
        if pretrained_user_emb is not None:
            old_u = min(n_users, pretrained_user_emb.shape[0])
            self.user_emb.weight.data[:old_u].copy_(torch.from_numpy(pretrained_user_emb[:old_u]))
            
        if pretrained_item_emb is not None:
            old_i = min(n_items, pretrained_item_emb.shape[0])
            self.item_emb.weight.data[:old_i].copy_(torch.from_numpy(pretrained_item_emb[:old_i]))
            
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

# 3. Huấn luyện LightGCN (MLflow)

EMB_DIM = 128
LAYERS = 3
EPOCHS = 10
LR = 0.01
DECAY = 1e-4 


user_vec_path = os.path.join(ARTIFACTS_DIR, "user_vectors.npy")
item_vec_path = os.path.join(ARTIFACTS_DIR, "item_vectors.npy")
pretrained_user, pretrained_item = None, None

if os.path.exists(user_vec_path) and os.path.exists(item_vec_path):
    print("Phát hiện Pre-trained Vectors! Kích hoạt chế độ Fine-Tuning (Warm-Start)...")
    pretrained_user = np.load(user_vec_path)
    pretrained_item = np.load(item_vec_path)
    EPOCHS = 2 
    LR = 0.0005 
else:
    print("Train mô hình từ đầu (Cold-Start)...")

model = LightGCN(num_users, num_items, emb_dim=EMB_DIM, layers=LAYERS, 
                 pretrained_user_emb=pretrained_user, 
                 pretrained_item_emb=pretrained_item).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Thiết lập đích danh MLflow Experiment để tránh lỗi trên Python Script Task
mlflow.set_experiment("/Users/truongtrinhdac03@gmail.com/LightGCN_Recommendation")

print(f"Bắt đầu huấn luyện LightGCN trên {device} với {EPOCHS} Epochs...")
with mlflow.start_run(run_name="LightGCN_Training"):
    mlflow.log_params({"emb_dim": EMB_DIM, "layers": LAYERS, "epochs": EPOCHS, "lr": LR, "weight_decay": DECAY})
    
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
            
            # L2 Regularization tính riêng cho base embedding
            u_base = model.user_emb(u)
            i_base = model.item_emb(i)
            j_base = model.item_emb(j)
            reg_loss = DECAY * (u_base.norm(2).pow(2) + i_base.norm(2).pow(2) + j_base.norm(2).pow(2)) / float(len(u))
            
            loss = bpr_loss + reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        mlflow.log_metric("total_loss", avg_loss, step=epoch)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

# 4. Trích xuất Vectors & TF-IDF/SVD cho Cold-Start


# 1. Trích xuất LightGCN Embeddings
model.eval()
with torch.no_grad():
    final_u, final_i = model(norm_adj_tensor)
    user_vectors = final_u.cpu().numpy().astype(np.float32)
    item_vectors = final_i.cpu().numpy().astype(np.float32)

faiss.normalize_L2(user_vectors)
faiss.normalize_L2(item_vectors)

# 2. Xây dựng TF-IDF + SVD (Cold-Start) dựa trên Metadata
print("\nBắt đầu xây dựng Content-based Model (TF-IDF + SVD) cho Cold-Start...")
all_texts = []
for msid, meta in item_meta.items():
    artist = meta.get('artist_name', '').strip().lower()
    track = meta.get('track_name', '').strip().lower()
    text = f"{artist} {artist} {track}".strip()
    all_texts.append(text if text else "unknown")

print(f"Tổng số bài hát để train TF-IDF: {len(all_texts):,}")

# Khởi tạo và huấn luyện TF-IDF (tham số lấy y nguyên từ bản gốc)
tfidf = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 1),
    max_features=20000,
    sublinear_tf=True, min_df=5
)
tfidf_matrix = tfidf.fit_transform(all_texts)

# Khởi tạo và huấn luyện TruncatedSVD giảm chiều về 64
svd_dim = min(64, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
svd = TruncatedSVD(n_components=svd_dim, random_state=42, n_iter=5)
dense_content_matrix = svd.fit_transform(tfidf_matrix).astype(np.float32)

faiss.normalize_L2(dense_content_matrix)
print("Hoàn tất huấn luyện TF-IDF và SVD!")

# 5. Lưu toàn bộ Artifacts 


print(f"\nLưu file tạm vào {TMP_DIR}...")
np.save(os.path.join(TMP_DIR, "user_vectors.npy"), user_vectors)
np.save(os.path.join(TMP_DIR, "item_vectors.npy"), item_vectors)

# Lưu TF-IDF và SVD Models
joblib.dump(tfidf, os.path.join(TMP_DIR, "tfidf_model.pkl"))
joblib.dump(svd, os.path.join(TMP_DIR, "svd_64_model.pkl"))

print(f"Copy files sang Unity Catalog: {ARTIFACTS_DIR}...")
files_to_copy = [
    "user_vectors.npy", 
    "item_vectors.npy", 
    "tfidf_model.pkl", 
    "svd_64_model.pkl"
]

for file_name in files_to_copy:
    src = os.path.join(TMP_DIR, file_name)
    dst = os.path.join(ARTIFACTS_DIR, file_name)
    shutil.copy(src, dst)

print("✅ Đã lưu toàn bộ Artifacts thành công!")

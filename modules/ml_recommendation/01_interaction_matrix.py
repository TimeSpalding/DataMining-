# Databricks notebook source
# MAGIC %md
# MAGIC # Tạo Ma trận tương tác (Interaction Matrix) cho Hệ thống Gợi ý
# MAGIC **Mục tiêu:** Sử dụng bảng `silver_unified_logs` (nhật ký đã làm sạch) để tạo tập train và test theo thời gian (Temporal split).
# MAGIC **Output:** Các dictionary mapping và file ma trận sparse.

# COMMAND ----------

import numpy as np
import scipy.sparse as sp
import joblib
import pandas as pd
from pyspark.sql import functions as F
import os
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


# 1. Tạo một Volume mới (sân bãi hợp lệ) bằng Spark SQL nếu nó chưa tồn tại
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.recommender_artifacts")

# 2. Khai báo đường dẫn mới trỏ thẳng vào Volume vừa tạo
ARTIFACTS_DIR = "/Volumes/workspace/default/recommender_artifacts"

# 3. Serverless hoàn toàn cho phép os.makedirs hoạt động bên trong /Volumes/
import os
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Kéo dữ liệu từ Databricks

# COMMAND ----------

# Lấy log tương tác từ silver table
silver_logs_df = spark.table("default.silver_unified_logs")

# Lọc các tương tác rác nếu cần, và chuyển về Pandas để thao tác ma trận
# Nếu dữ liệu siêu lớn, có thể dùng groupby count trên PySpark rồi mới collect
pdf = silver_logs_df.select("user_id", "recording_msid", "timestamp").toPandas()

# Tạo metadata (ánh xạ ID -> Tên bài hát/Ca sĩ)
meta_df = silver_logs_df.select("recording_msid", "track_name", "artist_name").dropDuplicates(["recording_msid"]).toPandas()
item_meta = meta_df.set_index("recording_msid").to_dict('index')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Temporal Split (Tách Train/Test theo thời gian)

# COMMAND ----------

# Sắp xếp log theo thời gian
pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
pdf = pdf.sort_values(by=['user_id', 'timestamp'])

# Giữ lại 20% tương tác MỚI NHẤT của từng user làm tập Test
def split_user(group):
    n = len(group)
    n_test = int(n * 0.2)
    if n_test == 0:
        return group, pd.DataFrame(columns=group.columns)
    return group.iloc[:-n_test], group.iloc[-n_test:]

train_list, test_list = [], []
for uid, group in pdf.groupby('user_id'):
    train_part, test_part = split_user(group)
    train_list.append(train_part)
    test_list.append(test_part)

train_pdf = pd.concat(train_list)
test_pdf = pd.concat(test_list)

print(f"Total interactions: {len(pdf)}")
print(f"Train set: {len(train_pdf)}")
print(f"Test set: {len(test_pdf)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Ánh xạ ID và Xây dựng Ma trận Sparse

# COMMAND ----------

# Lấy danh sách user và item duy nhất từ tập Train
unique_users = train_pdf['user_id'].unique()
unique_items = train_pdf['recording_msid'].unique()

user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {i: j for j, i in enumerate(unique_items)}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {j: i for i, j in item2idx.items()}

num_users = len(user2idx)
num_items = len(item2idx)

print(f"Users: {num_users}, Items: {num_items}")

# Hàm tạo ma trận sparse (dựa trên số lần nghe / hoặc simply binary 1/0)
def build_sparse_matrix(df, u2i, i2i, shape):
    # Lọc bỏ user/item không có trong tập train
    df = df[df['user_id'].isin(u2i.keys()) & df['recording_msid'].isin(i2i.keys())]
    
    # Tính số lần play
    counts = df.groupby(['user_id', 'recording_msid']).size().reset_index(name='plays')
    
    rows = counts['user_id'].map(u2i).values
    cols = counts['recording_msid'].map(i2i).values
    data = np.ones(len(rows)) # Đơn giản hóa: chuyển thành Implicit Feedback (1/0)
    
    return sp.csr_matrix((data, (rows, cols)), shape=shape)

train_matrix = build_sparse_matrix(train_pdf, user2idx, item2idx, (num_users, num_items))
test_matrix  = build_sparse_matrix(test_pdf,  user2idx, item2idx, (num_users, num_items))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Lưu Artifacts

# COMMAND ----------

# Lưu cấu trúc mapping
mappings = {
    'user2idx': user2idx,
    'item2idx': item2idx,
    'idx2user': idx2user,
    'idx2item': idx2item,
    'item_meta': item_meta
}
joblib.dump(mappings, os.path.join(ARTIFACTS_DIR, "index_mappings.pkl"))

# Lưu ma trận sparse
sp.save_npz(os.path.join(ARTIFACTS_DIR, "train_matrix.npz"), train_matrix)
sp.save_npz(os.path.join(ARTIFACTS_DIR, "test_matrix.npz"), test_matrix)

print(f"Đã lưu ma trận và mapping vào: {ARTIFACTS_DIR}")

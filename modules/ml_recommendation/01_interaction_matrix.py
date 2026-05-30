import numpy as np
import scipy.sparse as sp
import joblib
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import os

spark = SparkSession.builder.getOrCreate()

spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.recommender_artifacts")

ARTIFACTS_DIR = "/Volumes/workspace/default/recommender_artifacts"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

CONFIG = {
    'min_interactions': 20,
    'recency_halflife': 60, 
    'test_ratio': 0.2,
    'min_items_for_split': 5
}

# 1. Đọc dữ liệu và tính toán trọng số (Recency Weighting)


# Lấy log tương tác từ silver table
silver_logs_df = spark.table("default.silver_unified_logs")

# Chuyển timestamp sang unix (giây) để dễ tính toán khoảng thời gian
df = silver_logs_df.withColumn("ts_unix", F.unix_timestamp("timestamp"))
df = df.filter(F.col("ts_unix").isNotNull())

# Lấy mốc thời gian lớn nhất (global_max_ts) trong toàn bộ dữ liệu
global_max_ts = df.agg(F.max("ts_unix")).collect()[0][0]
print(f"Global max timestamp: {pd.Timestamp(global_max_ts, unit='s')}")

# Aggregation: đếm số lần play và lấy last_ts cho từng cặp (user, item)
agg_df = df.groupBy("user_id", "recording_msid").agg(
    F.count("*").alias("play_count"),
    F.max("ts_unix").alias("last_ts")
)

# Tính toán trọng số tương tác (Weighted Interaction)
# weight = play_count * exp(-days_ago / halflife)
agg_df = agg_df.withColumn(
    "days_ago", (F.lit(global_max_ts) - F.col("last_ts")) / 86400.0
).withColumn(
    "recency", F.exp(-F.col("days_ago") / CONFIG['recency_halflife'])
).withColumn(
    "weight", F.col("play_count") * F.col("recency")
)

# 2. Lọc Min Interactions


# Tính số lượng item mỗi user đã nghe, và số lượng user đã nghe mỗi item
user_counts = agg_df.groupBy("user_id").count().withColumnRenamed("count", "user_ic_cnt")
item_counts = agg_df.groupBy("recording_msid").count().withColumnRenamed("count", "item_uc_cnt")

# Lọc bỏ các user và item có tổng số lần xuất hiện < min_interactions
filtered_df = agg_df.join(user_counts, on="user_id") \
                    .join(item_counts, on="recording_msid") \
                    .filter((F.col("user_ic_cnt") >= CONFIG['min_interactions']) & 
                            (F.col("item_uc_cnt") >= CONFIG['min_interactions']))

# 3. Temporal Train/Test Split


# Xếp hạng item theo last_ts cho mỗi user (từ cũ nhất đến mới nhất)
window_spec = Window.partitionBy("user_id").orderBy(F.col("last_ts").asc())
split_df = filtered_df.withColumn("rank", F.row_number().over(window_spec)) \
                      .withColumn("total_items", F.count("*").over(Window.partitionBy("user_id")))

# Xác định item nào thuộc tập test (20% item nghe gần nhất)
split_df = split_df.withColumn(
    "n_test",
    F.when(
        F.col("total_items") >= CONFIG['min_items_for_split'], 
        F.greatest(F.lit(1), (F.col("total_items") * CONFIG['test_ratio']).cast("int"))
    ).otherwise(0)
)

split_df = split_df.withColumn(
    "is_test",
    F.col("rank") > (F.col("total_items") - F.col("n_test"))
)

# Kéo dữ liệu về Pandas để xây dựng ma trận Sparse
pdf = split_df.select("user_id", "recording_msid", "weight", "is_test").toPandas()

# 4. Ánh xạ ID và Xây dựng Ma trận Sparse


# Xây dựng danh sách index mapping
unique_users = pdf['user_id'].unique()
unique_items = pdf['recording_msid'].unique()

user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {i: j for j, i in enumerate(unique_items)}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {j: i for i, j in item2idx.items()}

num_users = len(user2idx)
num_items = len(item2idx)
print(f"Users: {num_users:,}, Items: {num_items:,}")

def build_sparse_matrix(df, shape):
    rows = df['user_id'].map(user2idx).values
    cols = df['recording_msid'].map(item2idx).values
    data = df['weight'].values
    return sp.csr_matrix((data, (rows, cols)), shape=shape, dtype=np.float32)

train_pdf = pdf[~pdf['is_test']]
test_pdf  = pdf[pdf['is_test']]

train_matrix = build_sparse_matrix(train_pdf, (num_users, num_items))
test_matrix  = build_sparse_matrix(test_pdf, (num_users, num_items))

print(f"Train set: {train_matrix.nnz:,} interactions")
print(f"Test set: {test_matrix.nnz:,} interactions")

# 5. Lưu Artifacts


import shutil

# Lấy metadata cho các item còn lại
meta_df = silver_logs_df.select("recording_msid", "track_name", "artist_name").dropDuplicates(["recording_msid"]).toPandas()
meta_df = meta_df[meta_df['recording_msid'].isin(item2idx.keys())]
item_meta = meta_df.set_index("recording_msid").to_dict('index')

# Lưu cấu trúc mapping
mappings = {
    'user2idx': user2idx,
    'item2idx': item2idx,
    'idx2user': idx2user,
    'idx2item': idx2item,
    'item_meta': item_meta
}

tmp_dir = "/tmp/recommender_artifacts"
os.makedirs(tmp_dir, exist_ok=True)

tmp_mappings = os.path.join(tmp_dir, "index_mappings.pkl")
tmp_train = os.path.join(tmp_dir, "train_matrix.npz")
tmp_test = os.path.join(tmp_dir, "test_matrix.npz")

print("Đang lưu file tạm tại /tmp...")
joblib.dump(mappings, tmp_mappings)
sp.save_npz(tmp_train, train_matrix)
sp.save_npz(tmp_test, test_matrix)

print(f"Đang copy file sang Volume: {ARTIFACTS_DIR}...")
shutil.copy(tmp_mappings, os.path.join(ARTIFACTS_DIR, "index_mappings.pkl"))
shutil.copy(tmp_train, os.path.join(ARTIFACTS_DIR, "train_matrix.npz"))
shutil.copy(tmp_test, os.path.join(ARTIFACTS_DIR, "test_matrix.npz"))

print(f"Đã lưu ma trận và mapping thành công vào: {ARTIFACTS_DIR}")


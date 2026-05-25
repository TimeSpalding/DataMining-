# Databricks notebook source
# MAGIC %md
# MAGIC # Phân cụm người dùng (User Persona Clustering)
# MAGIC **Mục tiêu:** Sử dụng bảng `gold_user_features` để gom cụm người dùng thành các Persona khác nhau dựa trên thói quen nghe nhạc (tần suất, mức độ đa dạng nghệ sĩ/thể loại, tỷ lệ nghe đêm).
# MAGIC **Output:** Cập nhật lại nhãn Persona cho từng user.

# COMMAND ----------

# Import thư viện cần thiết
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Đọc dữ liệu từ bảng Gold

# COMMAND ----------

# Đọc bảng gold_user_features từ Databricks Catalog
# Lưu ý: Thay đổi "default" thành tên schema của bạn nếu cần
gold_features_df = spark.table("default.gold_user_features")

# Xóa các dòng có giá trị null
gold_features_df = gold_features_df.dropna()

display(gold_features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Tiền xử lý & Chuẩn hóa (Scaling)

# COMMAND ----------

# Các cột feature sẽ dùng để gom cụm
feature_cols = [
    "total_listens", 
    "daily_listen_rate", 
    "night_listen_ratio", 
    "artist_diversity", 
    "track_diversity", 
    "tenure_days"
]

# 1. Gom các cột thành 1 vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df_assembled = assembler.transform(gold_features_df)

# 2. Chuẩn hóa dữ liệu (StandardScaler)
scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Giảm chiều dữ liệu (PCA) - Phục vụ trực quan hóa

# COMMAND ----------

# Giảm xuống 3 chiều để trực quan hóa
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

explained_variance = pca_model.explainedVariance.sum()
print(f"Tổng phương sai được giữ lại với 3 chiều PCA: {explained_variance * 100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gom cụm K-Means

# COMMAND ----------

# Khởi tạo mô hình K-Means (Giả sử số cụm tối ưu k=4)
k_optimal = 4
kmeans = KMeans(featuresCol="pca_features", predictionCol="cluster", k=k_optimal, seed=42)
kmeans_model = kmeans.fit(df_pca)

# Gán nhãn cụm cho dữ liệu
df_clustered = kmeans_model.transform(df_pca)

# Đánh giá Silhouette Score
evaluator = ClusteringEvaluator(featuresCol="pca_features", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(df_clustered)
print(f"Silhouette Score (k={k_optimal}): {silhouette:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Đặt tên Persona và Lưu kết quả

# COMMAND ----------

# Mapping tên cụm (Giả định sau khi phân tích tâm cụm)
persona_mapping = F.create_map([
    F.lit(0), F.lit("Explorer (Thích khám phá)"),
    F.lit(1), F.lit("Loyalist (Trung thành)"),
    F.lit(2), F.lit("Night Owl (Cú đêm)"),
    F.lit(3), F.lit("Casual (Nghe ngẫu hứng)")
])

# Thêm cột tên Persona
final_clustering_df = df_clustered.withColumn("persona_name", persona_mapping[F.col("cluster")])

# Chọn các cột cần thiết để lưu lại dưới dạng bảng (View hoặc Table)
output_df = final_clustering_df.select("user_id", "cluster", "persona_name")

# Ghi đè vào bảng persona_results trên Databricks
output_df.write.mode("overwrite").saveAsTable("default.user_persona_results")

print("Đã lưu kết quả phân cụm thành công vào bảng default.user_persona_results")

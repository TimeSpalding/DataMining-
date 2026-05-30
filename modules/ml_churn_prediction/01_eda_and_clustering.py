from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


gold_features_df = spark.table("default.gold_user_features")

gold_features_df = gold_features_df.dropna()

display(gold_features_df)

#Tiền xử lý & Chuẩn hóa (Scaling)


feature_cols = [
    "total_listens", 
    "daily_listen_rate", 
    "night_listen_ratio", 
    "artist_diversity", 
    "track_diversity", 
    "tenure_days"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features")
df_assembled = assembler.transform(gold_features_df)

scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withStd=True, withMean=True)
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)




# Giảm xuống 3 chiều để trực quan hóa
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")
pca_model = pca.fit(df_scaled)
df_pca = pca_model.transform(df_scaled)

explained_variance = pca_model.explainedVariance.sum()
print(f"Tổng phương sai được giữ lại với 3 chiều PCA: {explained_variance * 100:.2f}%")

# 4. Gom cụm K-Means

k_optimal = 4
kmeans = KMeans(featuresCol="pca_features", predictionCol="cluster", k=k_optimal, seed=42)
kmeans_model = kmeans.fit(df_pca)

# Gán nhãn cụm cho dữ liệu
df_clustered = kmeans_model.transform(df_pca)

# Đánh giá Silhouette Score
evaluator = ClusteringEvaluator(featuresCol="pca_features", predictionCol="cluster", metricName="silhouette", distanceMeasure="squaredEuclidean")
silhouette = evaluator.evaluate(df_clustered)
print(f"Silhouette Score (k={k_optimal}): {silhouette:.4f}")

# 5. Đặt tên Persona và Lưu kết quả


# Mapping tên cụm (Giả định sau khi phân tích tâm cụm)
persona_mapping = F.create_map([
    F.lit(0), F.lit("Explorer (Thích khám phá)"),
    F.lit(1), F.lit("Loyalist (Trung thành)"),
    F.lit(2), F.lit("Night Owl (Cú đêm)"),
    F.lit(3), F.lit("Casual (Nghe ngẫu hứng)")
])

final_clustering_df = df_clustered.withColumn("persona_name", persona_mapping[F.col("cluster")])

output_df = final_clustering_df.select("user_id", "cluster", "persona_name")

output_df.write.mode("overwrite").saveAsTable("default.user_persona_results")

print("Đã lưu kết quả phân cụm thành công vào bảng default.user_persona_results")

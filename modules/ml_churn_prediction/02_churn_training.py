# Databricks notebook source
# MAGIC %md
# MAGIC # Huấn luyện mô hình Dự đoán Rời bỏ (Churn Prediction)
# MAGIC **Mục tiêu:** Sử dụng dữ liệu từ `gold_user_features` và bảng lịch sử nghe nhạc để dán nhãn Churn (Time-Window). Sau đó huấn luyện Random Forest và dùng **MLflow** để tracking.
# MAGIC **Output:** Mô hình Random Forest được register trên MLflow và bảng dự đoán rủi ro churn.

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import datetime
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Đọc dữ liệu và Dán nhãn Time-Window

# COMMAND ----------

gold_features_df = spark.table("default.gold_user_features").dropna()
silver_logs_df = spark.table("default.silver_unified_logs")

# Giả sử chúng ta chia Time-Window: 
# Quá khứ (tính features): Trước ngày 2026-03-01
# Tương lai (xác định label): Từ 2026-03-01 đến 2026-03-31
# Ai không có dòng log nào trong tháng 3 -> Churn (1), có log -> Active (0)

# 1. Tìm các user có hoạt động trong khoảng thời gian tương lai
future_logs = silver_logs_df.filter(
    (F.col("timestamp") >= "2026-03-01") & 
    (F.col("timestamp") < "2026-04-01")
).select("user_id").distinct().withColumn("is_active", F.lit(1))

# 2. Join với gold_features (features tính từ quá khứ) để tạo label
labeled_data = gold_features_df.join(future_logs, on="user_id", how="left")

# Điền null bằng 0 (nghĩa là không hoạt động -> Churn = 1)
labeled_data = labeled_data.fillna({"is_active": 0})
labeled_data = labeled_data.withColumn("label", F.when(F.col("is_active") == 0, 1).otherwise(0))

print(f"Tổng số User: {labeled_data.count()}")
labeled_data.groupBy("label").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chuẩn bị dữ liệu Train/Test

# COMMAND ----------

feature_cols = [
    "total_listens", 
    "daily_listen_rate", 
    "night_listen_ratio", 
    "artist_diversity", 
    "track_diversity", 
    "tenure_days"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
ml_data = assembler.transform(labeled_data).select("user_id", "features", "label")

# Chia tập Train 80% / Test 20%
train_df, test_df = ml_data.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Huấn luyện với MLflow Tracking

# COMMAND ----------

# Bật autologging của MLflow cho PySpark ML
mlflow.pyspark.ml.autolog()

experiment_name = "/Users/truongtrinhdac03@gmail.com/Churn_Prediction_RF"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="RandomForest_TimeWindow"):
    # Cấu hình siêu tham số
    num_trees = 100
    max_depth = 7
    
    # Khởi tạo mô hình
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", 
                                numTrees=num_trees, maxDepth=max_depth, seed=42)
    
    # Train
    model = rf.fit(train_df)
    
    # Đánh giá trên tập test
    predictions = model.transform(test_df)
    
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    
    # Ghi nhận log thủ công thêm (Autolog đã lưu param nhưng có thể log thêm)
    mlflow.log_metric("test_auc", auc)
    print(f"Test AUC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Dự đoán toàn bộ và xuất rủi ro (Risk Percent)

# COMMAND ----------

# Dự đoán trên toàn bộ tập dữ liệu (để lấy ra % rủi ro cho Dashboard)
final_predictions = model.transform(ml_data)

# Cột probability chứa vector [prob_0, prob_1]. Ta lấy prob_1 (xác suất Churn)
extract_prob = F.udf(lambda v: float(v[1]), "double")
final_results = final_predictions.withColumn("churn_risk_percent", extract_prob("probability") * 100)

# Phân loại rủi ro (Risk Level)
final_results = final_results.withColumn(
    "risk_level",
    F.when(F.col("churn_risk_percent") >= 70, "HIGH")
     .when(F.col("churn_risk_percent") >= 40, "MEDIUM")
     .otherwise("LOW")
)

# Lưu kết quả xuống Databricks
output_df = final_results.select("user_id", "churn_risk_percent", "risk_level")
output_df.write.mode("overwrite").saveAsTable("default.churn_predictions")

print("Đã lưu kết quả dự đoán Churn vào bảng default.churn_predictions")

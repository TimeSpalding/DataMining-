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

# 1. Đọc dữ liệu và Dán nhãn Time-Window


gold_features_df = spark.table("default.gold_user_features").dropna()
silver_logs_df = spark.table("default.silver_unified_logs")


# 1. Tìm các user có hoạt động trong khoảng thời gian tương lai
future_logs = silver_logs_df.filter(
    (F.col("timestamp") >= "2026-03-01") & 
    (F.col("timestamp") < "2026-04-01")
).select("user_id").distinct().withColumn("is_active", F.lit(1))

# 2. Join với gold_features (features tính từ quá khứ) để tạo label
labeled_data = gold_features_df.join(future_logs, on="user_id", how="left")

labeled_data = labeled_data.fillna({"is_active": 0})
labeled_data = labeled_data.withColumn("label", F.when(F.col("is_active") == 0, 1).otherwise(0))

print(f"Tổng số User: {labeled_data.count()}")
labeled_data.groupBy("label").count().show()


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

train_df, test_df = ml_data.randomSplit([0.8, 0.2], seed=42)

# 3. Huấn luyện với MLflow Tracking

experiment_name = "/Users/truongtrinhdac03@gmail.com/Churn_Prediction_RF"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="RandomForest_TimeWindow"):
    # Cấu hình siêu tham số
    num_trees = 100
    max_depth = 7

    mlflow.log_param("num_trees", num_trees)
    mlflow.log_param("max_depth", max_depth)
    
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
    
    spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.mlflow_tmp")
    
    mlflow.spark.log_model(
        model, 
        "spark-model", 
        dfs_tmpdir="/Volumes/workspace/default/mlflow_tmp"
)

# 4. Dự đoán toàn bộ và xuất rủi ro (Risk Percent)


# Dự đoán trên toàn bộ tập dữ liệu (để lấy ra % rủi ro cho Dashboard)
final_predictions = model.transform(ml_data)

extract_prob = F.udf(lambda v: float(v[1]), "double")
final_results = final_predictions.withColumn("churn_risk_percent", extract_prob("probability") * 100)

# Phân loại rủi ro (Risk Level)
final_results = final_results.withColumn(
    "risk_level",
    F.when(F.col("churn_risk_percent") >= 70, "HIGH")
     .when(F.col("churn_risk_percent") >= 40, "MEDIUM")
     .otherwise("LOW")
)
output_df = final_results.select("user_id", "churn_risk_percent", "risk_level")
output_df.write.mode("overwrite").saveAsTable("default.churn_predictions")

print("Đã lưu kết quả dự đoán Churn vào bảng default.churn_predictions")

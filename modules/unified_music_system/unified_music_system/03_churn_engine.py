"""
03_churn_engine.py — LAYER 2B: "Nhà Máy B"
============================================
Đầu vào : data/outputs/clean_data/ (từ Layer 1)
Đầu ra  : data/outputs/web_dashboard_data_v2.csv
           → user_id, churn_risk_percent, persona_label, dominant_genre,
             total_listens, daily_listen_rate, tenure_days, ...

Thuật toán:
  - Time-window split (tránh data leakage)
  - Feature Engineering nâng cao (tenure, night ratio, diversity, ...)
  - PySpark Random Forest Classifier
  - Xuất churn_risk_percent (%) cho từng user

Chạy: python 03_churn_engine.py
"""
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OUTPUT_DIR, SPARK_CONFIG, CHURN_CUTOFF_DATE,
    CHURN_WINDOW_DAYS, CHURN_RF_TREES, CHURN_RF_DEPTH, CHURN_FEATURE_COLS,
    CHURN_CSV, RICH_PROFILE_CSV,
)


def build_spark():
    builder = SparkSession.builder.appName("ChurnEngine")
    for k, v in SPARK_CONFIG.items():
        if k != "appName":
            builder = builder.config(k, v)
    spark = builder.getOrCreate()
    # spark.sparkContext.setLogLevel("WARN")
    return spark


def run(spark: SparkSession):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("LAYER 2B — CHURN PREDICTION ENGINE")
    print("=" * 60)

    # ── 1. Đọc dữ liệu từ Tầng Silver (Databricks) ────────────────────────
    print(f"\n[1/5] Đọc dữ liệu từ bảng default.silver_unified_logs...")
    
    # "Hút" thẳng dữ liệu sạch từ Delta Table thay vì CSV
    df_raw = spark.read.table("default.silver_unified_logs")

    df_clean = (
        df_raw
        # Bảng Silver đã xử lý sẵn kiểu Timestamp chuẩn, ta chỉ cần đổi tên cột cho khớp logic
        .withColumnRenamed("timestamp", "ts")
        .dropna(subset=["ts", "user_id", "recording_msid"])
        .withColumn("hour", F.hour("ts"))
        .withColumn("is_night",
            F.when((F.col("hour") >= 22) | (F.col("hour") <= 5), 1).otherwise(0)
        )
    )

    # ── 2. Time-window split ─────────────────────────────────────────────────
    print(f"\n[2/5] Time-window split | cutoff: {CHURN_CUTOFF_DATE}...")
    df_obs  = df_clean.filter(F.col("ts") <= CHURN_CUTOFF_DATE)  # Quan sát
    df_pred = df_clean.filter(F.col("ts") >  CHURN_CUTOFF_DATE)  # Dự đoán

    # ── 3. Feature Engineering từ cửa sổ quan sát ───────────────────────────
    print("\n[3/5] Feature Engineering...")
    user_features = df_obs.groupBy("user_id").agg(
        F.max("ts").alias("last_listen_date"),
        F.min("ts").alias("first_listen_date"),
        F.count("recording_msid").alias("total_listens"),
        F.approx_count_distinct("artist_name", rsd=0.05).alias("unique_artists"),
        F.sum("is_night").alias("night_listens"),
    )

    def safe_ratio(num, den):
        return F.col(num) / F.when(F.col(den) > 0, F.col(den)).otherwise(1)

    user_profile = (
        user_features
        .withColumn("tenure_days",
            F.datediff("last_listen_date", "first_listen_date")
        )
        .withColumn("tenure_days",
            F.when(F.col("tenure_days") == 0, 1).otherwise(F.col("tenure_days"))
        )
        .withColumn("daily_listen_rate",
            F.round(safe_ratio("total_listens", "tenure_days"), 2)
        )
        .withColumn("night_listen_ratio",
            F.round(safe_ratio("night_listens", "total_listens"), 4)
        )
        .withColumn("artist_diversity",
            F.round(safe_ratio("unique_artists", "total_listens"), 4)
        )
    )

    # ── 4. Tạo nhãn churn từ cửa sổ dự đoán ─────────────────────────────────
    print(f"\n[4/5] Tạo nhãn Churn (active = nghe trong {CHURN_WINDOW_DAYS} ngày tới)...")
    active_users = (
        df_pred.select("user_id").distinct()
        .withColumn("is_active", F.lit(1))
    )
    final_df = (
        user_profile
        .join(active_users, on="user_id", how="left")
        .withColumn("is_churn",
            F.when(F.col("is_active").isNull(), 1).otherwise(0)
        )
        .drop("is_active", "last_listen_date", "first_listen_date")
    )
    # final_df.cache()

    churn_dist = final_df.groupBy("is_churn").count().collect()
    for row in churn_dist:
        label = "Churn" if row["is_churn"] == 1 else "Active"
        print(f"  {label}: {row['count']:,}")

    # ── 5. Huấn luyện Random Forest ──────────────────────────────────────────
    print("\n[5/5] Huấn luyện Random Forest...")
    feature_cols = [c for c in CHURN_FEATURE_COLS if c in final_df.columns]
    assembler = VectorAssembler(
        inputCols=feature_cols, outputCol="features", handleInvalid="skip"
    )
    ml_data = assembler.transform(final_df).withColumnRenamed("is_churn", "label")
    train_data, test_data = ml_data.randomSplit([0.8, 0.2], seed=42)

    rf = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=CHURN_RF_TREES, maxDepth=CHURN_RF_DEPTH,
        seed=42,
    )
    rf_model = rf.fit(train_data)

    # Đánh giá trên test set
    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = evaluator.evaluate(rf_model.transform(test_data))
    print(f"  ✅ AUC-ROC (time-window): {auc:.4f}")

    # Dự đoán trên toàn bộ users
    all_preds = rf_model.transform(ml_data)
    final_results = all_preds.withColumn(
        "churn_risk_percent",
        F.round(vector_to_array(F.col("probability"))[1] * 100, 2)
    )

    # Phân loại mức độ rủi ro
    web_data = final_results.select(
        "user_id", "total_listens", "daily_listen_rate",
        "night_listen_ratio", "artist_diversity", "tenure_days",
        "churn_risk_percent",
    ).withColumn("churn_tier",
        F.when(F.col("churn_risk_percent") >= 70, "HIGH")
         .when(F.col("churn_risk_percent") >= 40, "MEDIUM")
         .otherwise("LOW")
    )

    # Ghép thêm persona_label và dominant_genre từ rich_user_profile nếu có
    rich_path = os.path.join(OUTPUT_DIR, "rich_user_profile")
    try:
        rich_df = spark.read.option("header", "true").csv(rich_path)
        if "user_type" in rich_df.columns and "Dominant_Genre" in rich_df.columns:
            rich_slim = rich_df.select(
                "user_id",
                F.col("user_type").alias("persona_label"),
                F.col("Dominant_Genre").alias("dominant_genre"),
            )
            web_data = web_data.join(rich_slim, "user_id", "left")
            print("  Ghép persona_label & dominant_genre: OK")
    except Exception:
        print("  [SKIP] rich_user_profile chưa có — chạy 02 trước để ghép persona")

    # Xuất ra CSV
    pandas_df = web_data.toPandas()
    out_path  = CHURN_CSV
    pandas_df.to_csv(out_path, index=False)
    print(f"  ✅ Đã lưu: {out_path}")
    print(f"  Rows: {len(pandas_df):,} | Churn HIGH: {(pandas_df['churn_tier']=='HIGH').sum():,}")

    print("\n✅ Layer 2B — Churn Engine hoàn tất!")


if __name__ == "__main__":
    spark = build_spark()
    try:
        run(spark)
    finally:
        spark.stop()

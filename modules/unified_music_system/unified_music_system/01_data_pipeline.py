# """
# 01_data_pipeline.py — LAYER 1: Data Lake
# =========================================
# Đọc toàn bộ file CSV/XLSX thô, làm sạch và chuẩn hóa thành
# một Spark DataFrame duy nhất, sau đó lưu ra CSV sạch.

# Chạy: python 01_data_pipeline.py
# """
# import os, glob
# import pandas as pd
# from pyspark.sql import SparkSession
# from pyspark.sql import functions as F
# from pyspark.sql.types import StringType
# from pyspark import StorageLevel

# import sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import (
#     DATA_DIR, OUTPUT_DIR, SPARK_CONFIG, REQUIRED_COLS, TIMESTAMP_FORMAT
# )


# # ─────────────────────────── HELPERS ────────────────────────────────────────

# def build_spark() -> SparkSession:
#     builder = SparkSession.builder
#     for k, v in SPARK_CONFIG.items():
#         if k == "appName":
#             builder = builder.appName(v)
#         else:
#             builder = builder.config(k, v)
#     spark = builder.getOrCreate()
#     spark.sparkContext.setLogLevel("WARN")
#     return spark


# def load_mixed_data(spark: SparkSession, folder_path: str):
#     """Đọc *.csv (Spark native) và *.xlsx (Pandas→Spark) từ một thư mục."""
#     print(f"  → Loading: {folder_path}")
#     csv_files = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
#     xlsx_files = glob.glob(os.path.join(folder_path, "**", "*.xlsx"), recursive=True)

#     frames = []

#     # CSV — đọc native Spark, tự inferSchema
#     if csv_files:
#         df_csv = (
#             spark.read
#             .option("header", "true")
#             .option("inferSchema", "false")
#             .option("quote", '"')
#             .option("escape", '"')
#             .option("multiLine", "true")
#             .csv(csv_files)
#         )
#         frames.append(df_csv)

#     # XLSX — đọc qua Pandas rồi convert
#     for fpath in xlsx_files:
#         fname = os.path.basename(fpath)
#         if fname.startswith("~"):
#             continue
#         try:
#             pdf = pd.read_excel(fpath, dtype=str, engine="openpyxl")
#             pdf = pdf.loc[:, ~pdf.columns.str.contains("^Unnamed")]
#             frames.append(spark.createDataFrame(pdf))
#             print(f"    [XLSX] {fname} ({len(pdf):,} rows)")
#         except Exception as e:
#             print(f"    [WARN] {fname}: {e}")

#     if not frames:
#         raise ValueError(f"Không tìm thấy file nào trong: {folder_path}")

#     result = frames[0]
#     for f in frames[1:]:
#         result = result.unionByName(f, allowMissingColumns=True)
#     return result


# # ─────────────────────────── MAIN PIPELINE ───────────────────────────────────

# def run_pipeline(spark: SparkSession) -> None:
#     os.makedirs(OUTPUT_DIR, exist_ok=True)

#     print("\n" + "=" * 60)
#     print("LAYER 1 — DATA PIPELINE: Nạp & Làm Sạch Dữ Liệu")
#     print("=" * 60)

#     # 1. Nạp dữ liệu thô ────────────────────────────────────────────────────
#     print("\n[1/4] Đọc dữ liệu thô...")
#     df_raw = load_mixed_data(spark, DATA_DIR).repartition(200)
#     print(f"  Tổng dòng raw: {df_raw.count():,}")
#     print(f"  Columns: {df_raw.columns}")

#     # 2. Kiểm tra cột bắt buộc ───────────────────────────────────────────────
#     print("\n[2/4] Kiểm tra cột bắt buộc...")
#     missing = [c for c in REQUIRED_COLS if c not in df_raw.columns]
#     if missing:
#         raise ValueError(f"Dữ liệu thiếu cột: {missing}")

#     # Đếm null
#     null_exprs = [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in REQUIRED_COLS]
#     null_counts = df_raw.agg(*null_exprs).collect()[0].asDict()
#     for col_name, n in null_counts.items():
#         status = "OK" if n == 0 else f"⚠  {n:,} null"
#         print(f"  {col_name}: {status}")

#     # 3. Làm sạch & chuẩn hóa ───────────────────────────────────────────────
#     print("\n[3/4] Làm sạch dữ liệu...")

#     # Chỉ giữ cột cần thiết + các cột bổ sung nếu có
#     optional_cols = ["skip", "duration"]
#     keep_cols = REQUIRED_COLS + [c for c in optional_cols if c in df_raw.columns]
#     df_sel = df_raw.select(*[F.col(c).cast(StringType()) for c in keep_cols])

#     df_clean = (
#         df_sel
#         # Parse timestamp — thử format chuẩn trước, fallback sang tự động
#         .withColumn("ts_parsed",
#             F.coalesce(
#                 F.to_timestamp(F.col("timestamp"), TIMESTAMP_FORMAT),
#                 F.to_timestamp(F.col("timestamp")),
#             )
#         )
#         # Drop dòng thiếu trường quan trọng
#         .dropna(subset=["ts_parsed", "user_id", "recording_msid", "artist_name"])
#         # Chuẩn hóa text
#         .withColumn("user_id",         F.trim(F.col("user_id")))
#         .withColumn("recording_msid",  F.trim(F.col("recording_msid")))
#         .withColumn("track_name",      F.trim(F.col("track_name")))
#         .withColumn("artist_name",     F.trim(F.col("artist_name")))
#         # Đổi lại timestamp đã parse
#         .withColumn("timestamp",       F.date_format("ts_parsed", "yyyy-MM-dd HH:mm:ss"))
#         .drop("ts_parsed")
#         # Loại duplicate hoàn toàn
#         .dropDuplicates(["user_id", "recording_msid", "timestamp"])
#     )

#     # Ép kiểu optional columns
#     if "skip" in df_clean.columns:
#         df_clean = df_clean.withColumn("skip", F.col("skip").cast("int"))
#     if "duration" in df_clean.columns:
#         df_clean = df_clean.withColumn("duration", F.col("duration").cast("double"))

#     df_clean.persist(StorageLevel.MEMORY_AND_DISK)
#     n_clean = df_clean.count()
#     print(f"  Dòng sau làm sạch: {n_clean:,}")
#     print(f"  Users unique: {df_clean.select('user_id').distinct().count():,}")
#     print(f"  Tracks unique: {df_clean.select('recording_msid').distinct().count():,}")

#     # 4. Xuất ra CSV sạch ────────────────────────────────────────────────────
#     print("\n[4/4] Lưu dữ liệu sạch...")
#     out_path = os.path.join(OUTPUT_DIR, "clean_data")
#     df_clean.coalesce(10).write.mode("overwrite").option("header", "true").csv(out_path)
#     print(f"  ✅ Đã lưu: {out_path}/")

#     # In thống kê nhanh
#     print("\n── Thống kê hoạt động người dùng ──")
#     user_stats = df_clean.groupBy("user_id").agg(F.count("*").alias("plays"))
#     q = user_stats.approxQuantile("plays", [0.25, 0.5, 0.75, 0.9], 0.01)
#     print(f"  P25={q[0]:.0f} | Median={q[1]:.0f} | P75={q[2]:.0f} | P90={q[3]:.0f}")

#     df_clean.unpersist()
#     print("\n✅ Layer 1 — Data Pipeline hoàn tất!")


# if __name__ == "__main__":
#     spark = build_spark()
#     try:
#         run_pipeline(spark)
#     finally:
#         spark.stop()

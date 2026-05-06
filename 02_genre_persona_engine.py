"""
02_genre_persona_engine.py — LAYER 2A: "Nhà Máy A"
=====================================================
Đầu vào : data/outputs/clean_data/ (từ Layer 1)
Đầu ra  :
  - user_taste_profile.csv    (Gu thể loại của từng user)
  - artist_genre_profile.csv  (Nhãn thể loại của từng nghệ sĩ)
  - rich_user_profile.csv     (Persona + Genre gộp)

Thuật toán:
  - Feature Engineering (Shannon Entropy, Session Analysis, Time Ratios)
  - Outlier capping (p1–p99) + 3-sigma filter
  - SVD (Singular Value Decomposition) cho dimensionality reduction
  - KMeans / BisectingKMeans / GMM (chọn tốt nhất theo Silhouette)
  - FP-Growth → Association rules
  - LSH + Label Propagation → Genre Classification
  
Chạy: python 02_genre_persona_engine.py
"""
import os, json, pickle, time
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import FloatType, DoubleType
from pyspark import StorageLevel
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, 
    CountVectorizer, MinHashLSH,
)
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, DenseMatrix
from pyspark.mllib.linalg import Vectors as OldVectors
from pyspark.mllib.linalg.distributed import RowMatrix

import sys
sys.path.insert(0, "/Workspace/Users/truongtd.b22kh130@stu.ptit.edu.vn/DataMining-/modules/unified_music_system")

from config import (
    OUTPUT_DIR, 
    RECOMMENDER_CONFIG, 
    CHURN_CUTOFF_DATE,
    LGCN_CONFIG
)

GENRES = ["POP", "HIPHOP", "EDM", "RNB", "ROCK"]
MIN_PLAYS_FOR_GENRE = 10


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def safe_ratio(num, den):
    return F.col(num) / F.when(F.col(den) > 0, F.col(den)).otherwise(1)


def cap_outliers(df, columns, lo=0.01, hi=0.99):
    valid = [c for c in columns if c in df.columns]
    if not valid:
        return df
    qs = df.approxQuantile(valid, [lo, hi], 0.01)
    for c, q in zip(valid, qs):
        if q and len(q) == 2 and None not in q:
            df = df.withColumn(
                c,
                F.when(F.col(c) < q[0], q[0])
                 .when(F.col(c) > q[1], q[1])
                 .otherwise(F.col(c))
            )
    return df


def save_csv(df, name, coalesce_n=1):
    path = os.path.join(OUTPUT_DIR, name)
    df.coalesce(coalesce_n).write.mode("overwrite").option("header", "true").csv(path)
    print(f" Saved: {path}/")


def compute_svd(df_features, feature_cols, k_components=10):
    """
    Compute SVD on the feature matrix using RowMatrix which is optimized for large data.
    
    Parameters:
    -----------
    df_features : Spark DataFrame
        DataFrame with features
    feature_cols : list
        List of feature column names
    k_components : int
        Number of singular vectors to compute
    
    Returns:
    --------
    tuple: (svd_result, transformed_df)
        - svd_result: RowMatrix SVD result
        - transformed_df: DataFrame with SVD-transformed features
    """
    print(f"  Computing SVD with {k_components} components...")
    
    # Convert features to RowMatrix
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vec = assembler.transform(df_features).select("user_id", "features")
    
    # Convert to RDD of Vectors for RowMatrix
    feature_rdd = df_vec.rdd.map(lambda row: OldVectors.fromML(row.features))
    row_matrix = RowMatrix(feature_rdd)
    
    # Compute SVD
    svd = row_matrix.computeSVD(k_components, computeU=True)
    
    # Transform the data to lower-dimensional space
    # U * s = projected points
    U = svd.U  # RowMatrix of left singular vectors
    s = svd.s  # singular values
    V = svd.V  # right singular vectors (DenseMatrix)
    
    # Project data to SVD space: X * V = U * diag(s)
    # Which is more efficient to compute
    projected = row_matrix.multiply(V)
    projected_rows = projected.rows.zipWithIndex().map(lambda x: (x[1], x[0]))
    
    # Convert back to DataFrame
    projected_df = projected_rows.map(lambda x: (int(x[0]), tuple(x[1].toArray().tolist()))).toDF(["index", "svd_features"])
    
    # Add user_id back
    result_df = df_vec.rdd.zipWithIndex().map(lambda x: (x[1], x[0].user_id)).toDF(["index", "user_id"]).join(projected_df, "index")
    
    # Convert array to vector
    result_df = result_df.withColumn("svd_features_vec", F.udf(lambda x: Vectors.dense(x), "vector")("svd_features"))
    
    # Calculate explained variance ratio
    total_var = np.sum(s ** 2)
    explained_var_ratio = (s ** 2) / total_var
    cumulative_var = np.cumsum(explained_var_ratio)
    
    optimal_k = np.searchsorted(cumulative_var, 0.90)[0] + 1
    optimal_k = min(optimal_k, k_components)
    
    print(f"  SVD explained variance: {cumulative_var[optimal_k-1]*100:.1f}% with {optimal_k} components")
    print(f"  Singular values: {s[:5]}...")
    
    # Keep only top optimal_k components for efficient storage
    keep_cols = [f"svd_{i}" for i in range(optimal_k)]
    
    # Extract individual SVD components as columns
    for i in range(optimal_k):
        result_df = result_df.withColumn(f"svd_{i}", F.col("svd_features")[i])
    
    result_df = result_df.select("user_id", *keep_cols)
    
    return svd, result_df, optimal_k, explained_var_ratio[:optimal_k]


def compute_svd_optimized(df_features, feature_cols, variance_threshold=0.90, max_components=15):
    """
    Optimized SVD computation with automatic component selection based on explained variance.
    """
    print(f"  Computing SVD with automatic component selection (target variance: {variance_threshold*100}%)...")
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_vec = assembler.transform(df_features).select("user_id", "features")
    
    # Cache to avoid recomputation
    df_vec.persist(StorageLevel.MEMORY_AND_DISK)
    
    # Convert to RDD for RowMatrix
    feature_rdd = df_vec.rdd.map(lambda row: OldVectors.fromML(row.features))
    row_matrix = RowMatrix(feature_rdd)
    
    # Step 1: Compute SVD with max components first to understand variance distribution
    svd_full = row_matrix.computeSVD(max_components, computeU=False)
    s = svd_full.s
    
    # Compute explained variance
    total_var = np.sum(s ** 2)
    explained_var = (s ** 2) / total_var
    cumulative_var = np.cumsum(explained_var)
    
    # Determine optimal number of components
    k_optimal = np.searchsorted(cumulative_var, variance_threshold) + 1
    k_optimal = min(k_optimal, max_components, len(feature_cols))
    
    print(f"  SVD variance analysis:")
    for i in range(min(10, len(cumulative_var))):
        print(f"    Component {i+1}: {explained_var[i]*100:.1f}% (cumulative: {cumulative_var[i]*100:.1f}%)")
    print(f"  Selected {k_optimal} components explaining {cumulative_var[k_optimal-1]*100:.1f}% variance")
    
    # Step 2: Compute SVD with optimal components (with U for transformation)
    svd = row_matrix.computeSVD(k_optimal, computeU=True)
    
    # Project data to lower dimension: X * V_k = U_k * diag(s_k)
    V = svd.V  # Right singular vectors
    projected_matrix = row_matrix.multiply(V)
    
    # Convert to DataFrame
    projected_rows = projected_matrix.rows.zipWithIndex()
    
    def create_features_vector(row_idx, vector):
        return (row_idx, vector.toArray().tolist())
    
    projected_data = projected_rows.map(lambda x: create_features_vector(x[1], x[0]))
    projected_df = projected_data.toDF(["index", "svd_features_array"])
    
    # Add user_id back
    user_ids = df_vec.rdd.zipWithIndex().map(lambda x: (x[1], x[0].user_id)).toDF(["index", "user_id"])
    result_df = user_ids.join(projected_df, "index")
    
    # Convert array to individual columns for better performance
    for i in range(k_optimal):
        result_df = result_df.withColumn(f"svd_{i}", F.col("svd_features_array")[i])
    
    result_df = result_df.select("user_id", *[f"svd_{i}" for i in range(k_optimal)])
    
    # Clean up
    df_vec.unpersist()
    
    return result_df, k_optimal, cumulative_var[k_optimal-1]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(spark: SparkSession):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("LAYER 2A — GENRE + PERSONA ENGINE (SVD OPTIMIZED)")
    print("=" * 60)

    # --- 1. Đọc dữ liệu từ Tầng Silver (Databricks) -------------------
    print("\n[1/8] Đọc dữ liệu từ bảng music_ai_workspace.default.silver_unified_logs...")

    df_raw = spark.table("music_ai_workspace.default.silver_unified_logs").sample(fraction=0.05, seed=42)

    df_proc = (
        df_raw
        .withColumnRenamed("timestamp", "ts")
        .withColumn("track_name", F.col("recording_msid"))
        .dropna(subset=["ts", "user_id", "track_name", "artist_name"])
        .withColumn("hour", F.hour("ts"))
        .withColumn("date", F.to_date("ts"))
        .withColumn("unix_ts", F.unix_timestamp("ts"))
    )

    n_rows = df_proc.count()
    print(f"  Records: {n_rows:,}")

    # ── 2. Feature Engineering ──────────────────────────────────────────────
    print("\n[2/8] Feature Engineering...")

    # 2a. Base aggregations
    base_exprs = [
        F.count("*").alias("total_listens"),
        F.approx_count_distinct("track_name",  rsd=0.05).alias("unique_tracks"),
        F.approx_count_distinct("artist_name", rsd=0.05).alias("unique_artists"),
        F.approx_count_distinct("date",        rsd=0.05).alias("active_days"),
        F.sum(F.when((F.col("hour") >= 22) | (F.col("hour") <= 4),  1).otherwise(0)).alias("night_listens"),
        F.sum(F.when((F.col("hour") >= 5)  & (F.col("hour") <= 11), 1).otherwise(0)).alias("morning_listens"),
        F.sum(F.when((F.col("hour") >= 12) & (F.col("hour") <= 17), 1).otherwise(0)).alias("afternoon_listens"),
        F.sum(F.when((F.col("hour") >= 18) & (F.col("hour") <= 21), 1).otherwise(0)).alias("evening_listens"),
        F.avg("hour").alias("avg_listen_hour"),
        F.stddev("hour").alias("hour_std"),
    ]
    if "skip" in df_proc.columns:
        base_exprs.append(F.avg(F.col("skip").cast("int")).alias("skip_rate"))
    if "duration" in df_proc.columns:
        base_exprs += [
            F.avg(F.col("duration").cast("double")).alias("avg_duration"),
        ]

    user_base = df_proc.groupBy("user_id").agg(*base_exprs)
    if "skip_rate" not in user_base.columns:
        user_base = user_base.withColumn("skip_rate", F.lit(0.0))

    # 2b. Session analysis (30-min timeout)
    w_sess = Window.partitionBy("user_id").orderBy("unix_ts")
    df_sess = (
        df_proc
        .withColumn("prev_ts",        F.lag("unix_ts").over(w_sess))
        .withColumn("time_gap",       F.col("unix_ts") - F.col("prev_ts"))
        .withColumn("is_new_session", F.when(
            F.col("prev_ts").isNull() | (F.col("time_gap") > 1800), 1
        ).otherwise(0))
        .withColumn("session_id",     F.sum("is_new_session").over(w_sess))
    )
    sess_metrics = df_sess.groupBy("user_id", "session_id").agg(
        F.count("*").alias("tracks_per_session"),
        F.sum("time_gap").alias("session_duration"),
    )
    user_session = sess_metrics.groupBy("user_id").agg(
        F.avg("tracks_per_session").alias("avg_tracks_per_session"),
        F.stddev("tracks_per_session").alias("std_tracks_per_session"),
        F.avg("session_duration").alias("avg_session_duration_seconds"),
        F.count("session_id").alias("total_sessions"),
    ).fillna(0)

    # 2c. Shannon Entropy (Artist)
    user_total = user_base.select("user_id", F.col("total_listens").alias("_total"))
    artist_entropy = (
        df_proc.groupBy("user_id", "artist_name").agg(F.count("*").alias("cnt"))
        .join(user_total, "user_id")
        .withColumn("p", F.col("cnt") / F.col("_total"))
        .withColumn("plogp", F.col("p") * F.log2(F.col("p")))
        .groupBy("user_id").agg((-F.sum("plogp")).alias("artist_diversity"))
    )
    time_entropy = (
        df_proc.groupBy("user_id", "hour").agg(F.count("*").alias("cnt"))
        .join(user_total, "user_id")
        .withColumn("p", F.col("cnt") / F.col("_total"))
        .withColumn("plogp", F.col("p") * F.log2(F.col("p")))
        .groupBy("user_id").agg((-F.sum("plogp")).alias("time_entropy"))
    )

    # 2d. Join tất cả
    user_features = (
        user_base
        .join(artist_entropy, "user_id", "left")
        .join(time_entropy,   "user_id", "left")
        .join(user_session,   "user_id", "left")
        .fillna(0)
        .withColumn("daily_listen_rate",  safe_ratio("total_listens",   "active_days"))
        .withColumn("night_listen_ratio", safe_ratio("night_listens",   "total_listens"))
        .withColumn("morning_ratio",      safe_ratio("morning_listens", "total_listens"))
        .withColumn("afternoon_ratio",    safe_ratio("afternoon_listens","total_listens"))
        .withColumn("evening_ratio",      safe_ratio("evening_listens", "total_listens"))
        .withColumn("log_total_listens",  F.log1p("total_listens"))
        .fillna(0)
    )
    print(f"  Feature engineering hoàn tất: {user_features.count():,} users")

    # ── 3. Outlier Handling ─────────────────────────────────────────────────
    print("\n[3/8] Outlier Handling...")
    FEATURE_COLS = [c for c in [
        "log_total_listens", "daily_listen_rate",
        "avg_tracks_per_session", "avg_session_duration_seconds",
        "artist_diversity", "time_entropy",
        "night_listen_ratio", "total_sessions",
        "skip_rate",
    ] if c in user_features.columns]

    user_features = cap_outliers(user_features, FEATURE_COLS)
    stats = user_features.select(
        F.mean(F.log1p("total_listens")).alias("m"),
        F.stddev(F.log1p("total_listens")).alias("s"),
    ).collect()[0]
    thresh = stats["m"] + 3 * stats["s"]
    n_before = user_features.count()
    user_features = user_features.filter(F.log1p(F.col("total_listens")) <= thresh)
    n_after = user_features.count()
    print(f"  Loại bỏ outliers: {n_before - n_after:,} users ({(n_before-n_after)/n_before*100:.1f}%)")

    # ── 4. Scale + SVD (Optimized for Large Data) ─────────────────────────────────
    print("\n[4/8] SVD Dimensionality Reduction...")
    
    # Scale features first
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="scaled_features", withMean=True, withStd=True)
    
    df_assembled = assembler.transform(user_features)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled).select("user_id", "scaled_features")
    
    # Convert scaled features to columns for SVD
    df_with_cols = df_scaled.select("user_id", *[F.col("scaled_features")[i].alias(f"f_{i}") for i in range(len(FEATURE_COLS))])
    
    # Apply SVD
    svd_df, n_components, explained_variance = compute_svd_optimized(
        df_with_cols, 
        [f"f_{i}" for i in range(len(FEATURE_COLS))],
        variance_threshold=0.90,
        max_components=min(len(FEATURE_COLS), 15)
    )
    
    print(f"  SVD reduced dimensions from {len(FEATURE_COLS)} to {n_components}")
    print(f"  Explained variance: {explained_variance*100:.1f}%")

    # ── 5. Clustering on SVD-reduced data ───────────────────────────────────
    print("\n[5/8] Clustering on SVD components...")
    
    # Prepare SVD components vector
    svd_cols = [f"svd_{i}" for i in range(n_components)]
    svd_assembler = VectorAssembler(inputCols=svd_cols, outputCol="svd_vector")
    df_svd_vector = svd_assembler.transform(svd_df).select("user_id", "svd_vector")
    
    # Choose best clustering algorithm and K using silhouette score
    best_algo, best_sil, best_k, predictions = None, -1.0, 4, None
    evaluator = ClusteringEvaluator(featuresCol="svd_vector", metricName="silhouette")
    
    # Test K values from 3 to 7
    for k in range(3, 8):
        # KMeans
        kmeans = KMeans(k=k, featuresCol="svd_vector", predictionCol="cluster", seed=42, maxIter=20)
        kmeans_model = kmeans.fit(df_svd_vector)
        kmeans_pred = kmeans_model.transform(df_svd_vector)
        kmeans_sil = evaluator.evaluate(kmeans_pred)
        
        # BisectingKMeans (often faster and better for large data)
        bkm = BisectingKMeans(k=k, featuresCol="svd_vector", predictionCol="cluster", seed=42, maxIter=20)
        bkm_model = bkm.fit(df_svd_vector)
        bkm_pred = bkm_model.transform(df_svd_vector)
        bkm_sil = evaluator.evaluate(bkm_pred)
        
        print(f"  k={k}: KMeans silhouette={kmeans_sil:.4f}, BisectingKMeans silhouette={bkm_sil:.4f}")
        
        if kmeans_sil > best_sil:
            best_sil, best_k, best_algo, predictions = kmeans_sil, k, "KMeans", kmeans_pred
        if bkm_sil > best_sil:
            best_sil, best_k, best_algo, predictions = bkm_sil, k, "BisectingKMeans", bkm_pred
    
    print(f" Best clustering: {best_algo} with k={best_k}, silhouette={best_sil:.4f}")
    
    # Gán nhãn cluster → label ngôn ngữ tự nhiên
    cluster_stats = (
        predictions.join(user_features.select("user_id", *FEATURE_COLS), "user_id")
        .groupBy("cluster")
        .agg(
            F.avg("artist_diversity").alias("avg_diversity"),
            F.avg("night_listen_ratio").alias("avg_night"),
            F.avg("daily_listen_rate").alias("avg_daily"),
            F.avg("avg_tracks_per_session").alias("avg_session_depth"),
            F.count("user_id").alias("size"),
        )
        .orderBy("cluster")
        .collect()
    )

    label_map = {}
    for row in cluster_stats:
        cid = row["cluster"]
        parts = []
        parts.append("Active" if (row["avg_daily"] or 0) > 5 else "Casual")
        parts.append("Night Owl" if (row["avg_night"] or 0) > 0.3 else "Daytime")
        parts.append("Explorer" if (row["avg_diversity"] or 0) > 2.5 else "Loyalist")
        parts.append("Deep Session" if (row["avg_session_depth"] or 0) > 8 else "Light Session")
        label_map[cid] = " | ".join(parts)
    print(f"  Cluster labels: {label_map}")

    label_sdf = spark.createDataFrame(
        [(int(k), v) for k, v in label_map.items()], ["cluster", "user_type"]
    )
    clustering_results = (
        predictions.join(label_sdf, "cluster", "left")
        .join(user_features.select("user_id", *FEATURE_COLS), "user_id")
        .select("user_id", "cluster", "user_type", *FEATURE_COLS)
    )

    # ── 6. FP-Growth ─────────────────────────────────────────────────────────
    print("\n[6/8] FP-Growth Association Rules...")
    user_artists = (
        df_proc
        .groupBy("user_id")
        .agg(F.collect_set("artist_name").alias("artists"))
        .filter(F.size("artists") >= 3)
    )
    
    fp_model = FPGrowth(
        itemsCol="artists", minSupport=0.02, minConfidence=0.4,
    )
    fp_rules_spark_df = fp_model.fit(user_artists).associationRules

    if not fp_rules_spark_df.isEmpty():
        fp_rules_spark_df.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable("music_ai_workspace.default.gold_association_rules")
        print(f"Đã lưu Luật FP-Growth vào bảng gold_association_rules")

    # ── 7. Genre Classification (LSH + Label Propagation) ───────────────────
    print("\n[7/8] Genre Classification (LSH)...")

    GENRE_SEEDS = {
        "POP"    : ["taylor swift", "ed sheeran", "ariana grande", "billie eilish",
                    "dua lipa", "the weeknd", "harry styles", "olivia rodrigo"],
        "ROCK"   : ["radiohead", "nirvana", "foo fighters", "arctic monkeys",
                    "the strokes", "muse", "green day", "linkin park"],
        "HIPHOP" : ["kendrick lamar", "drake", "j. cole", "eminem",
                    "kanye west", "travis scott", "post malone", "21 savage"],
        "EDM"    : ["daft punk", "deadmau5", "avicii", "martin garrix",
                    "tiesto", "calvin harris", "skrillex", "kygo"],
        "RNB"    : ["frank ocean", "sza", "h.e.r.", "giveon",
                    "daniel caesar", "jhene aiko", "bryson tiller", "miguel"],
    }

    user_artist_counts = (
        df_proc.groupBy("user_id", "artist_name")
        .agg(F.count("*").alias("play_count"))
        .withColumn("artist_lower", F.lower(F.col("artist_name")))
    )

    seed_rows = []
    for genre, artists in GENRE_SEEDS.items():
        for artist in artists:
            seed_rows.append((artist, genre))
    seed_df = spark.createDataFrame(seed_rows, ["artist_lower", "seed_genre"])

    seeded = user_artist_counts.join(seed_df, "artist_lower", "inner")
    seed_users = (
        seeded.groupBy("user_id")
        .agg(
            F.sum(F.when(F.col("seed_genre") == "POP",    F.col("play_count")).otherwise(0)).alias("POP_plays"),
            F.sum(F.when(F.col("seed_genre") == "ROCK",   F.col("play_count")).otherwise(0)).alias("ROCK_plays"),
            F.sum(F.when(F.col("seed_genre") == "HIPHOP", F.col("play_count")).otherwise(0)).alias("HIPHOP_plays"),
            F.sum(F.when(F.col("seed_genre") == "EDM",    F.col("play_count")).otherwise(0)).alias("EDM_plays"),
            F.sum(F.when(F.col("seed_genre") == "RNB",    F.col("play_count")).otherwise(0)).alias("RNB_plays"),
        )
        .withColumn("total_seed_plays",
            F.col("POP_plays") + F.col("ROCK_plays") + F.col("HIPHOP_plays") +
            F.col("EDM_plays") + F.col("RNB_plays")
        )
        .filter(F.col("total_seed_plays") > 0)
    )

    GCOLS = [f"{g}_%" for g in GENRES]

    def ratio_col(plays_col, total_col):
        return F.round(F.col(plays_col) / F.when(F.col(total_col) > 0, F.col(total_col)).otherwise(1) * 100, 1)

    seed_labeled = (
        seed_users
        .withColumn("POP_%",    ratio_col("POP_plays",    "total_seed_plays"))
        .withColumn("ROCK_%",   ratio_col("ROCK_plays",   "total_seed_plays"))
        .withColumn("HIPHOP_%", ratio_col("HIPHOP_plays", "total_seed_plays"))
        .withColumn("EDM_%",    ratio_col("EDM_plays",    "total_seed_plays"))
        .withColumn("RNB_%",    ratio_col("RNB_plays",    "total_seed_plays"))
        .select("user_id", *GCOLS)
    )

    max_col = F.greatest(*[F.col(c) for c in GCOLS])
    user_taste_profile = (
        seed_labeled
        .withColumn("_mx", max_col)
        .withColumn("Dominant_Genre",
            F.when(F.col("POP_%")    == F.col("_mx"), "POP")
             .when(F.col("HIPHOP_%") == F.col("_mx"), "HIPHOP")
             .when(F.col("EDM_%")    == F.col("_mx"), "EDM")
             .when(F.col("RNB_%")    == F.col("_mx"), "RNB")
             .otherwise("ROCK")
        )
        .drop("_mx")
    )
    
    # Fill unlabeled users with propagated values
    unlabeled_users = df_proc.select("user_id").distinct().join(user_taste_profile, "user_id", "left_anti")
    default_genre = seed_labeled.select(*GCOLS).agg(*[F.avg(c).alias(c) for c in GCOLS]).collect()[0]
    default_vals = {c: float(default_genre[c] or 20.0) for c in GCOLS}
    
    propagated_full = unlabeled_users.withColumn("Dominant_Genre", F.lit("EXPLORER"))
    for c in GCOLS:
        propagated_full = propagated_full.withColumn(c, F.lit(default_vals[c]))
        
    user_taste_combined = user_taste_profile.unionByName(propagated_full, allowMissingColumns=True).distinct()
    print(f"User Taste Profile hoàn tất: {user_taste_combined.count():,} users")

    # Artist Genre Profile
    artist_genre_profile = (
        df_proc.join(user_taste_profile.select("user_id", *GCOLS), "user_id")
        .groupBy("artist_name").agg(
            F.count("user_id").alias("fan_count"),
            *[F.round(F.avg(c), 1).alias(c) for c in GCOLS]
        )
        .filter(F.col("fan_count") >= MIN_PLAYS_FOR_GENRE)
        .withColumn("_mx", F.greatest(*[F.col(c) for c in GCOLS]))
        .withColumn("Dominant_Genre",
            F.when(F.col("POP_%")    == F.col("_mx"), "POP")
             .when(F.col("HIPHOP_%") == F.col("_mx"), "HIPHOP")
             .when(F.col("EDM_%")    == F.col("_mx"), "EDM")
             .when(F.col("RNB_%")    == F.col("_mx"), "RNB")
             .otherwise("ROCK")
        )
        .drop("_mx")
    )
    print(f"  Artist Genre Profile: {artist_genre_profile.count():,} nghệ sĩ")

    # ── 8. Rich User Profile (Clustering + Genre) ────────────────────────────
    print("\n[8/8] Rich User Profile (Gộp Clustering + Genre)...")
    rich_profile = (
        clustering_results
        .join(user_taste_combined.select("user_id", *GCOLS, "Dominant_Genre"), "user_id", "left")
        .withColumn("profile_summary",
            F.concat(F.col("user_type"), F.lit(" | "), F.coalesce(F.col("Dominant_Genre"), F.lit("EXPLORER")))
        )
    )
    print(f"  Rich User Profile: {rich_profile.count():,} users")

    # ── 9. Lưu kết quả ──────────────────────────────────────────────────────
    print("\n[9/8] Lưu kết quả...")
    save_csv(user_taste_combined,  "user_taste_profile")
    save_csv(artist_genre_profile, "artist_genre_profile")
    save_csv(rich_profile,         "rich_user_profile")
    save_csv(clustering_results,   "user_clusters")
    
    # Save SVD components and loadings for future use
    svd_artifacts = {
        "n_components": n_components,
        "explained_variance": explained_variance,
        "feature_cols": FEATURE_COLS,
        "svd_components": svd_df.columns
    }
    
    with open(os.path.join(OUTPUT_DIR, "svd_artifacts.pkl"), "wb") as f:
        pickle.dump(svd_artifacts, f)

    # Save model artifacts
    model_out = os.path.join(OUTPUT_DIR, "model_artifacts.pkl")
    with open(model_out, "wb") as f:
        pickle.dump({
            "best_algorithm": best_algo,
            "optimal_k": best_k,
            "silhouette": best_sil,
            "cluster_labels": label_map,
            "feature_cols": FEATURE_COLS,
            "genres": GENRES,
            "svd_components": n_components,
            "svd_explained_variance": explained_variance,
        }, f)
    print(f" Saved: {model_out}")

    print("\n Layer 2A — Genre + Persona Engine hoàn tất!")


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("GenrePersonaEngine_SVD") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "400") \
        .config("spark.local.dir", "/tmp/spark_tmp") \
        .getOrCreate()
    
    run(spark)
    spark.stop()
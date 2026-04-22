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
  - PCA + BisectingKMeans / KMeans / GMM (chọn tốt nhất theo Silhouette)
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
    VectorAssembler, StandardScaler, PCA,
    CountVectorizer, MinHashLSH,
)
from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture
from pyspark.ml.evaluation import ClusteringEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.fpm import FPGrowth
from pyspark.ml.functions import vector_to_array

import sys
sys.path.insert(0, "/Workspace/Users/truongtrinhdac03@gmail.com/DataMining-/modules/unified_music_system")
from config import (
    OUTPUT_DIR, SPARK_CONFIG, GENRES,
    MIN_PLAYS_FOR_GENRE, CHURN_CUTOFF_DATE
)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def build_spark():
    builder = SparkSession.builder
    for k, v in SPARK_CONFIG.items():
        if k == "appName":
            builder = builder.appName("GenrePersonaEngine")
        else:
            builder = builder.config(k, v)
    builder = (
        builder
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "400")
        .config("spark.local.dir", "/tmp/spark_tmp")
    )
    spark = builder.getOrCreate()
    # spark.sparkContext.setLogLevel("WARN")
    # spark.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")
    return spark


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
    print(f"  ✅ Saved: {path}/")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def run(spark: SparkSession):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("LAYER 2A — GENRE + PERSONA ENGINE")
    print("=" * 60)

    # ── 1. Đọc dữ liệu từ Tầng Silver (Databricks) ────────────────────────
    print(f"\n[1/8] Đọc dữ liệu từ bảng default.silver_unified_logs...")
    
    # Hút dữ liệu từ Database
    df_raw = spark.read.table("default.silver_unified_logs")

    df_proc = (
        df_raw
        .withColumnRenamed("timestamp", "ts")
        # 🚨 VÁ LỖ HỔNG DỮ LIỆU: Mượn tạm msid làm tên bài hát
        .withColumn("track_name", F.col("recording_msid"))
        # Drop các giá trị null thực sự
        .dropna(subset=["ts", "user_id", "track_name", "artist_name"])
        .withColumn("hour",    F.hour("ts"))
        .withColumn("date",    F.to_date("ts"))
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
    # user_features.persist(StorageLevel.MEMORY_AND_DISK)
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
    # user_features.persist(StorageLevel.MEMORY_AND_DISK)

    # ── 4. PCA + Clustering ─────────────────────────────────────────────────
    print("\n[4/8] PCA + Clustering...")
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features", handleInvalid="skip")
    scaler    = StandardScaler(inputCol="raw_features", outputCol="scaled_features",
                               withMean=True, withStd=True)

    df_assembled = assembler.transform(user_features)
    scaler_model  = scaler.fit(df_assembled)
    df_scaled     = scaler_model.transform(df_assembled).select("user_id", "scaled_features")

    # PCA — tự chọn số chiều giải thích 90% variance
    pca_model_full = PCA(k=min(len(FEATURE_COLS), 8), inputCol="scaled_features", outputCol="pca_features")
    pca_result     = pca_model_full.fit(df_scaled)
    cumvar = np.cumsum(pca_result.explainedVariance.toArray())
    k_pca  = int(np.searchsorted(cumvar, 0.90)) + 1
    k_pca  = max(2, min(k_pca, len(FEATURE_COLS)))
    print(f"  PCA: {k_pca} chiều giải thích {cumvar[k_pca-1]*100:.1f}% variance")

    pca_model = PCA(k=k_pca, inputCol="scaled_features", outputCol="pca_features")
    pca_fit   = pca_model.fit(df_scaled)
    df_pca    = pca_fit.transform(df_scaled).select("user_id", "pca_features")
    # df_pca.cache()

    # Chọn K tốt nhất (range 3–7)
    best_algo, best_sil, best_k, predictions = None, -1.0, 4, None
    evaluator = ClusteringEvaluator(featuresCol="pca_features", metricName="silhouette")
    comparison = []

    for k in range(3, 8):
        for algo_name, AlgoCls, kwargs in [
            ("BisectingKMeans", BisectingKMeans, {"seed": 42}),
            ("KMeans",          KMeans,          {"seed": 42}),
        ]:
            model = AlgoCls(k=k, featuresCol="pca_features", predictionCol="cluster", **kwargs).fit(df_pca)
            preds = model.transform(df_pca)
            sil   = evaluator.evaluate(preds)
            comparison.append({"algorithm": algo_name, "k": k, "silhouette": sil})
            if sil > best_sil:
                best_sil, best_algo, best_k, predictions = sil, algo_name, k, preds
            print(f"    {algo_name} k={k}: silhouette={sil:.4f}")

    print(f"  ✅ Best: {best_algo} k={best_k} silhouette={best_sil:.4f}")

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
        parts.append(f"Deep Session" if (row["avg_session_depth"] or 0) > 8 else "Light Session")
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

    # ── 5. FP-Growth ─────────────────────────────────────────────────────────
    print("\n[5/8] FP-Growth Association Rules...")
    user_artists = (
        df_proc
        .groupBy("user_id")
        .agg(F.collect_set("artist_name").alias("artists"))
        .filter(F.size("artists") >= 3)
    )
    cv  = CountVectorizer(inputCol="artists", outputCol="artist_vec", minDF=5.0)
    cv_model = cv.fit(user_artists)
    user_artist_vec = cv_model.transform(user_artists)

    fp_model = FPGrowth(
        itemsCol="artists", minSupport=0.02, minConfidence=0.4,
    )
    fp_rules_df = fp_model.fit(user_artists).associationRules.toPandas()
    if not fp_rules_df.empty:
        fp_path = os.path.join(OUTPUT_DIR, "fp_growth_rules.csv")
        fp_rules_df.to_csv(fp_path, index=False)
        print(f"  FP-Growth: {len(fp_rules_df)} luật → {fp_path}")

    # ── 6. Genre Classification (LSH + Label Propagation) ───────────────────
    print("\n[6/8] Genre Classification (LSH)...")

    # Gán label giả dựa trên tên nghệ sĩ (seed artists)
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

    # Tính user-artist listen profile
    user_artist_counts = (
        df_proc.groupBy("user_id", "artist_name")
        .agg(F.count("*").alias("play_count"))
        .withColumn("artist_lower", F.lower(F.col("artist_name")))
    )

    # Label seed users
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

    # Label propagation via LSH (MinHash) cho users không có seed
    mh = MinHashLSH(inputCol="artist_vec", outputCol="hashes", numHashTables=5, seed=42)
    mh_model = mh.fit(user_artist_vec)
    mh_trans = mh_model.transform(user_artist_vec)

    labeled_vec = mh_trans.join(user_taste_profile.select("user_id", "Dominant_Genre"), "user_id", "inner")
    unlabeled_vec = mh_trans.join(user_taste_profile.select("user_id"), "user_id", "left_anti")

    approx_sim = mh_model.approxSimilarityJoin(
        unlabeled_vec.select("user_id", "artist_vec", "hashes").withColumnRenamed("user_id", "u_id"),
        labeled_vec.select("user_id", "artist_vec", "hashes", "Dominant_Genre"),
        threshold=0.8,
        distCol="distance",
    )

    propagated = (
        approx_sim
        .groupBy(F.col("datasetA.u_id").alias("user_id"))
        .agg(F.first("datasetB.Dominant_Genre").alias("Dominant_Genre"))
    )

    default_genre = seed_labeled.select(*GCOLS).agg(*[F.avg(c).alias(c) for c in GCOLS]).collect()[0]
    default_vals  = {c: float(default_genre[c] or 20.0) for c in GCOLS}

    propagated_full = propagated.select("user_id", "Dominant_Genre")
    for c in GCOLS:
        propagated_full = propagated_full.withColumn(c, F.lit(default_vals[c]))

    user_taste_combined = user_taste_profile.unionByName(propagated_full, allowMissingColumns=True).distinct()
    # user_taste_combined.persist(StorageLevel.MEMORY_AND_DISK)
    print(f"  User Taste Profile: {user_taste_combined.count():,} users")

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

    # ── 7. Rich User Profile (Clustering + Genre) ────────────────────────────
    print("\n[7/8] Rich User Profile (Gộp Clustering + Genre)...")
    rich_profile = (
        clustering_results
        .join(user_taste_combined.select("user_id", *GCOLS, "Dominant_Genre"), "user_id", "left")
        .withColumn("profile_summary",
            F.concat(F.col("user_type"), F.lit(" | "), F.coalesce(F.col("Dominant_Genre"), F.lit("UNKNOWN")))
        )
    )
    # rich_profile.persist()
    print(f"  Rich User Profile: {rich_profile.count():,} users")

    # ── 8. Lưu kết quả ──────────────────────────────────────────────────────
    print("\n[8/8] Lưu kết quả...")
    save_csv(user_taste_combined,  "user_taste_profile")
    save_csv(artist_genre_profile, "artist_genre_profile")
    save_csv(rich_profile,         "rich_user_profile")
    save_csv(clustering_results,   "user_clusters")

    # Lưu model artifacts
    model_out = os.path.join(OUTPUT_DIR, "model_artifacts.pkl")
    with open(model_out, "wb") as f:
        pickle.dump({
            "best_algorithm": best_algo,
            "optimal_k":      best_k,
            "silhouette":     float(best_sil),
            "cluster_labels": label_map,
            "feature_cols":   FEATURE_COLS,
            "genres":         GENRES,
            "algorithm_comparison": comparison,
        }, f)
    print(f"  ✅ Saved: {model_out}")

    # df_proc.unpersist()
    # user_features.unpersist()
    # rich_profile.unpersist()
    print("\n✅ Layer 2A — Genre + Persona Engine hoàn tất!")


if __name__ == "__main__":
    spark = build_spark()
    try:
        run(spark)
    finally:
        spark.stop()

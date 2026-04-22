"""
config.py — Cấu hình trung tâm cho toàn bộ Unified Music System
Chỉnh sửa file này là toàn bộ hệ thống tự điều chỉnh theo.
"""
import os

# ─────────────────────────── ĐƯỜNG DẪN ───────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "raw")        # Đặt file CSV/XLSX gốc vào đây
OUTPUT_DIR      = os.path.join(BASE_DIR, "data", "outputs")    # Layer 2 ghi kết quả vào đây
MODEL_DIR       = os.path.join(BASE_DIR, "data", "models")     # LightGCN checkpoint

# Layer 3 — Knowledge Base
KB_PATH         = os.path.join(BASE_DIR, "data", "mappings.db")

# Layer 2 — Output artifacts (các file CSV/NPY do 3 engine sinh ra)
CHURN_CSV       = os.path.join(OUTPUT_DIR, "web_dashboard_data_v2.csv")
USER_TASTE_CSV  = os.path.join(OUTPUT_DIR, "user_taste_profile.csv")
ARTIST_GENRE_CSV= os.path.join(OUTPUT_DIR, "artist_genre_profile.csv")
RICH_PROFILE_CSV= os.path.join(OUTPUT_DIR, "rich_user_profile.csv")
USER_VEC_NPY    = os.path.join(MODEL_DIR,  "user_vectors.npy")
ITEM_VEC_NPY    = os.path.join(MODEL_DIR,  "item_vectors.npy")
INDEX_MAPPINGS  = os.path.join(MODEL_DIR,  "index_mappings.pkl")

# ─────────────────────────── LAYER 1 — DATA PIPELINE ─────────────────────────
SPARK_CONFIG = {
    "appName"                       : "UnifiedMusicSystem",
    "spark.driver.memory"           : "8g",
    "spark.executor.memory"         : "8g",
    "spark.sql.adaptive.enabled"    : "true",
    "spark.sql.shuffle.partitions"  : "200",
    "spark.serializer"              : "org.apache.spark.serializer.KryoSerializer",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
}
# Cột bắt buộc phải có trong data gốc
REQUIRED_COLS = ["user_id", "recording_msid", "track_name", "artist_name", "timestamp"]
TIMESTAMP_FORMAT = "yyyy-MM-dd HH:mm:ss"

# ─────────────────────────── LAYER 2A — GENRE + PERSONA ENGINE ───────────────
GENRES = ["POP", "HIPHOP", "EDM", "RNB", "ROCK"]
CHURN_CUTOFF_DATE   = "2026-01-24 23:59:59"   # Cửa sổ thời gian: mốc cutoff
MIN_PLAYS_FOR_GENRE = 10                        # Artist cần ≥ 10 fan để có genre profile

# ─────────────────────────── LAYER 2B — CHURN ENGINE ─────────────────────────
CHURN_WINDOW_DAYS   = 14     # Không nghe > 14 ngày → label churn = 1
CHURN_RF_TREES      = 50
CHURN_RF_DEPTH      = 5
CHURN_FEATURE_COLS  = [
    "total_listens", "daily_listen_rate",
    "night_listen_ratio", "artist_diversity",
    "tenure_days",
]

# ─────────────────────────── LAYER 2C — LIGHTGCN ENGINE ──────────────────────
LGCN_CONFIG = {
    "emb_dim"          : 128,
    "n_layers"         : 3,
    "lr"               : 1e-3,
    "decay"            : 1e-3,
    "epochs"           : 20,
    "batch_size"       : 8192,
    "grad_clip"        : 1.0,
    "min_interactions" : 20,
    "chunk_size"       : 500_000,
    "recency_halflife" : 60,     # half-life decay (ngày)
    "neg_pool_size"    : 2_000_000,
}

# ─────────────────────────── LAYER 4 — RECOMMENDER ───────────────────────────
RECOMMENDER_CONFIG = {
    # Hybrid scoring
    "default_content_alpha"  : 0.25,   # trọng số TF-IDF vs LightGCN
    "genre_bonus_multiplier" : 1.20,   # +20% điểm nếu genre khớp user
    "churn_risk_threshold"   : 70.0,   # % churn risk → bật chế độ bảo vệ
    "churn_trending_weight"  : 0.6,    # trọng số trending khi user sắp churn
    "cold_threshold"         : 5,      # item < 5 interaction → cold item
    "mmr_lambda"             : 0.5,    # diversity (MMR reranking)
    "trending_decay_days"    : 30,     # half-life trending score
    "session_decay"          : 0.8,    # decay factor trong session
    "serendipity_default"    : 0.3,    # mức độ khám phá mặc định
}

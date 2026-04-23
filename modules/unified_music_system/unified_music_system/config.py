import os

# =============================================================================
# 1. CẤU HÌNH CƠ SỞ HẠ TẦNG (INFRASTRUCTURE)
# =============================================================================
# Đường dẫn gốc trong Workspace của bạn
USER_EMAIL = "truongtd.b22kh130@stu.ptit.edu.vn"
BASE_DIR = f"/Workspace/Users/{USER_EMAIL}/DataMining"

# Các thư mục con để lưu Model và Log
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Nơi lưu file Database SQLite cuối cùng (Để bạn tải về máy chạy Web)
KB_PATH = os.path.join(BASE_DIR, "music_knowledge.db")

# =============================================================================
# 2. CẤU HÌNH UNITY CATALOG (DATA WAREHOUSE)
# =============================================================================
CATALOG = "music_ai_workspace"
SCHEMA = "default"

# Định nghĩa tên các bảng để các module gọi cho đồng bộ
TABLE_SILVER_LOGS = f"{CATALOG}.{SCHEMA}.silver_unified_logs"
TABLE_GOLD_PERSONA = f"{CATALOG}.{SCHEMA}.gold_user_persona"
TABLE_GOLD_CHURN   = f"{CATALOG}.{SCHEMA}.gold_churn_predictions"

# =============================================================================
# 3. THAM SỐ MÔ HÌNH AI (HYPERPARAMETERS)
# =============================================================================
# Cấu hình LightGCN (Module 04)
LGCN_CONFIG = {
    "emb_dim": 128,
    "n_layers": 3,
    "lr": 1e-3,
    "epochs": 20,         # Tăng/giảm tùy theo thời gian bạn có
    "batch_size": 8192,
    "chunk_size": 500000, # Nén dữ liệu để tránh tràn RAM
    "neg_pool_size": 2000000
}

# Cấu hình Hệ thống Gợi ý lai (Module Recommender)
RECOMMENDER_CONFIG = {
    "default_content_alpha": 0.25,  # Trọng số cho TF-IDF (Content-based)
    "churn_risk_threshold": 70.0,  # Ngưỡng báo động User sắp bỏ app
    "cold_threshold": 5            # Số lần nghe tối thiểu để không bị coi là "người lạ"
}

# Cấu hình Churn (Module 03)
CHURN_WINDOW_DAYS = 14
CHURN_RF_TREES = 50
CHURN_CUTOFF_DATE = "2026-01-24 23:59:59"

print(f"✅ Đã nạp cấu hình hệ thống từ config.py (Catalog: {CATALOG})")
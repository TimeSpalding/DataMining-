"""
05_build_knowledge_base.py — LAYER 3: Knowledge Base
======================================================
Đầu vào : Tất cả CSV từ Layer 2 (outputs/)
Đầu ra  : data/mappings.db (SQLite)

Schema:
  users   → user_id, churn_risk, churn_tier, persona_label, dominant_genre,
             total_listens, daily_listen_rate, tenure_days,
             night_listen_ratio, artist_diversity
  artists → artist_name, dominant_genre, fan_count, POP_%, HIPHOP_%, EDM_%, RNB_%, ROCK_%
  items   → msid, track_name, artist_name, dominant_genre (join từ artists)

Index:
  Tất cả lookup fields đều được đánh index để query nhanh.

Chạy: python 05_build_knowledge_base.py
"""
import os, sqlite3, glob
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "/Workspace/Users/truongtd.b22kh130@stu.ptit.edu.vn/DataMining-/modules/unified_music_system")
from config import (
    OUTPUT_DIR, KB_PATH, MODEL_DIR,
    CHURN_CSV, ARTIST_GENRE_CSV, USER_TASTE_CSV, RICH_PROFILE_CSV,
)

def get_pandas_from_table(table_name: str) -> pd.DataFrame:
    """Đọc bảng Delta từ Unity Catalog và chuyển về Pandas để nạp vào SQLite."""
    print(f"  -> Đang trích xuất dữ liệu từ bảng: {table_name}")
    # Chỉ lấy dữ liệu từ bảng Spark, chuyển về Pandas
    return spark.table(table_name).toPandas()


def normalize_col(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """Rename cột nếu tên khác nhau giữa các file."""
    if old in df.columns and new not in df.columns:
        df = df.rename(columns={old: new})
    return df


# ──────────────────────────────────────────────────────────────────────────────

def build_users_table(conn: sqlite3.Connection):
    print("\n[1/4] Building table: users...")
    
    # 1. Đọc thẳng bảng kết quả cuối cùng của Module 03 (đã có cả Churn và Persona)
    users_df = get_pandas_from_table("music_ai_workspace.default.gold_churn_predictions")
    
    # 2. Chuẩn hóa tên cột cho khớp với giao diện Web của bạn
    users_df = normalize_col(users_df, "churn_risk_percent", "churn_risk")
    
    # 3. Ghi vào SQLite
    users_cols = [
        "user_id", "churn_risk", "churn_tier", "persona_label", 
        "total_listens", "daily_listen_rate", "tenure_days", "artist_diversity"
    ]
    # Lọc lại những cột thực sự tồn tại
    final_cols = [c for c in users_cols if c in users_df.columns]
    
    users_df[final_cols].to_sql("users", conn, if_exists="replace", index=False)
    
    # Tạo Index để Web search cho nhanh
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_uid ON users(user_id)")
    conn.commit()
    print(f"  ✅ Users table: {len(users_df):,} rows")


def build_artists_table(conn: sqlite3.Connection):
    print("\n[2/4] Building table: artists...")

    artist_path = os.path.join(OUTPUT_DIR, "artist_genre_profile")
    try:
        artist_df = read_spark_csv_dir(artist_path)
    except FileNotFoundError:
        # Fallback: đọc CSV trực tiếp
        artist_df = pd.read_csv(ARTIST_GENRE_CSV, low_memory=False)

    artist_df.columns = artist_df.columns.str.strip()
    artist_df = normalize_col(artist_df, "Dominant_Genre", "dominant_genre")

    # Đảm bảo cột tên nghệ sĩ lowercase để join nhất quán
    artist_df["artist_name_lower"] = artist_df["artist_name"].str.lower().str.strip()

    genre_cols = ["POP_%", "HIPHOP_%", "EDM_%", "RNB_%", "ROCK_%"]
    keep_cols = (
        ["artist_name", "artist_name_lower", "dominant_genre", "fan_count"]
        + [c for c in genre_cols if c in artist_df.columns]
    )
    artist_df[keep_cols].to_sql("artists", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artists_name  ON artists(artist_name_lower)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artists_genre ON artists(dominant_genre)")
    conn.commit()
    print(f"  ✅ artists table: {len(artist_df):,} rows")


def build_items_table(conn: sqlite3.Connection):
    print("\n[3/4] Building table: items...")
    pkl_path = os.path.join(MODEL_DIR, "index_mappings.pkl")
    if not os.path.exists(pkl_path):
        print("  [SKIP] index_mappings.pkl chưa có — chạy 04 trước.")
        return

    import pickle
    with open(pkl_path, "rb") as f:
        mappings = pickle.load(f)

    rows = []
    for msid, meta in mappings["item_meta"].items():
        rows.append({
            "msid":        msid,
            "track_name":  meta.get("track_name", ""),
            "artist_name": meta.get("artist_name", ""),
        })

    items_df = pd.DataFrame(rows)
    if items_df.empty:
        print("  [SKIP] item_meta rỗng.")
        return

    # Join dominant_genre từ artists
    artists_df = pd.read_sql("SELECT artist_name_lower, dominant_genre FROM artists", conn)
    items_df["artist_name_lower"] = items_df["artist_name"].str.lower().str.strip()
    items_df = items_df.merge(artists_df, on="artist_name_lower", how="left")
    items_df["dominant_genre"] = items_df["dominant_genre"].fillna("POP")
    items_df = items_df.drop(columns=["artist_name_lower"])

    items_df.to_sql("items", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_msid   ON items(msid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_artist ON items(artist_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_genre  ON items(dominant_genre)")
    conn.commit()
    print(f"  ✅ items table: {len(items_df):,} rows")


def build_stats_table(conn: sqlite3.Connection):
    """Bảng metadata tóm tắt để hiển thị dashboard nhanh."""
    print("\n[4/4] Building table: system_stats...")
    stats = {}

    try:
        r = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        stats["total_users"] = r[0]
    except Exception:
        stats["total_users"] = 0

    try:
        r = conn.execute("SELECT COUNT(*) FROM users WHERE churn_risk >= 70").fetchone()
        stats["high_churn_users"] = r[0]
    except Exception:
        stats["high_churn_users"] = 0

    try:
        r = conn.execute("SELECT COUNT(*) FROM artists").fetchone()
        stats["total_artists"] = r[0]
    except Exception:
        stats["total_artists"] = 0

    try:
        r = conn.execute("SELECT COUNT(*) FROM items").fetchone()
        stats["total_items"] = r[0]
    except Exception:
        stats["total_items"] = 0

    stats_df = pd.DataFrame([stats])
    stats_df.to_sql("system_stats", conn, if_exists="replace", index=False)
    conn.commit()
    print(f"  ✅ system_stats: {stats}")


# ──────────────────────────────────────────────────────────────────────────────

def run():
    # 🎯 ĐƯỜNG DẪN QUAN TRỌNG: Lưu vào Workspace để bạn có thể Download về máy
    # Thay 'truongtd.b22kh130' bằng đúng folder user của bạn trên Databricks
    user_path = "/Workspace/Users/truongtd.b22kh130@stu.ptit.edu.vn/DataMining"
    kb_path = f"{user_path}/music_knowledge.db"
    
    print(f"🚀 Bắt đầu đóng gói Knowledge Base tại: {kb_path}")
    
    # Tạo thư mục nếu chưa có
    import os
    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    
    # Kết nối SQLite
    conn = sqlite3.connect(kb_path)
    
    try:
        # Chạy lần lượt các xưởng sản xuất
        build_users_table(conn)    # Bạn đã sửa hàm này theo ý tôi ở tin nhắn trước chưa?
        build_artists_table(conn)
        build_items_table(conn)
        build_stats_table(conn)
        
        # Kiểm tra dung lượng file cuối cùng
        db_size = os.path.getsize(kb_path) / (1024 * 1024)
        print("\n" + "="*40)
        print(f"✅ THÀNH CÔNG!")
        print(f"📦 File: music_knowledge.db ({db_size:.2f} MB)")
        print(f"📍 Vị trí: {kb_path}")
        print("👉 Bây giờ hãy vào Workspace, click chuột phải vào file và chọn Download!")
        print("="*40)
        
    except Exception as e:
        print(f"❌ Thất bại: {e}")
    finally:
        conn.close()

# Kích hoạt
if __name__ == "__main__":
    run()

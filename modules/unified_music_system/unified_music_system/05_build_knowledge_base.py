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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    OUTPUT_DIR, KB_PATH, MODEL_DIR,
    CHURN_CSV, ARTIST_GENRE_CSV, USER_TASTE_CSV, RICH_PROFILE_CSV,
)


def read_spark_csv_dir(dir_path: str) -> pd.DataFrame:
    """Đọc thư mục CSV từ Spark (nhiều part-*.csv)."""
    if not os.path.isdir(dir_path):
        # Thử như file trực tiếp
        if os.path.isfile(dir_path):
            return pd.read_csv(dir_path, low_memory=False)
        raise FileNotFoundError(f"Không tìm thấy: {dir_path}")
    parts = glob.glob(os.path.join(dir_path, "part-*.csv"))
    if not parts:
        parts = glob.glob(os.path.join(dir_path, "*.csv"))
    if not parts:
        raise FileNotFoundError(f"Không có file CSV nào trong: {dir_path}")
    return pd.concat([pd.read_csv(p, low_memory=False) for p in parts], ignore_index=True)


def normalize_col(df: pd.DataFrame, old: str, new: str) -> pd.DataFrame:
    """Rename cột nếu tên khác nhau giữa các file."""
    if old in df.columns and new not in df.columns:
        df = df.rename(columns={old: new})
    return df


# ──────────────────────────────────────────────────────────────────────────────

def build_users_table(conn: sqlite3.Connection):
    print("\n[1/4] Building table: users...")

    # Đọc churn data
    churn_df = pd.read_csv(CHURN_CSV, low_memory=False)
    churn_df.columns = churn_df.columns.str.strip()

    # Chuẩn hóa tên cột
    churn_df = normalize_col(churn_df, "churn_risk_percent", "churn_risk")
    churn_df = normalize_col(churn_df, "user_type", "persona_label")
    churn_df = normalize_col(churn_df, "Dominant_Genre", "dominant_genre")

    # Đảm bảo các cột bắt buộc tồn tại
    required = {
        "user_id": "UNKNOWN", "churn_risk": 50.0,
        "churn_tier": "MEDIUM", "persona_label": "Unknown",
        "dominant_genre": "POP", "total_listens": 0,
        "daily_listen_rate": 0.0, "tenure_days": 0,
        "night_listen_ratio": 0.0, "artist_diversity": 0.0,
    }
    for col, default in required.items():
        if col not in churn_df.columns:
            churn_df[col] = default

    # Thêm persona & genre từ rich_profile nếu chưa có
    try:
        rich_path = os.path.join(OUTPUT_DIR, "rich_user_profile")
        rich_df   = read_spark_csv_dir(rich_path)
        rich_df.columns = rich_df.columns.str.strip()
        rich_df = normalize_col(rich_df, "user_type",     "persona_label")
        rich_df = normalize_col(rich_df, "Dominant_Genre","dominant_genre")

        merge_cols = ["user_id"] + [
            c for c in ["persona_label", "dominant_genre"]
            if c in rich_df.columns
        ]
        churn_df = churn_df.drop(
            columns=[c for c in ["persona_label", "dominant_genre"] if c in churn_df.columns],
            errors="ignore",
        )
        churn_df = churn_df.merge(rich_df[merge_cols].drop_duplicates("user_id"),
                                  on="user_id", how="left")
        for col, default in [("persona_label", "Unknown"), ("dominant_genre", "POP")]:
            churn_df[col] = churn_df[col].fillna(default)
        print("  Ghép rich_user_profile: OK")
    except Exception as e:
        print(f"  [SKIP] rich_user_profile: {e}")

    # Ghi vào SQLite
    users_cols = [
        "user_id", "churn_risk", "churn_tier", "persona_label", "dominant_genre",
        "total_listens", "daily_listen_rate", "tenure_days",
        "night_listen_ratio", "artist_diversity",
    ]
    final_cols = [c for c in users_cols if c in churn_df.columns]
    churn_df[final_cols].to_sql("users", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_uid ON users(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_churn ON users(churn_risk)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_genre ON users(dominant_genre)")
    conn.commit()
    print(f"  ✅ users table: {len(churn_df):,} rows")


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
    # Đảm bảo đường dẫn này khớp với config.py bạn đã sửa (BASE_DIR = "/dbfs/music_models")
    print(f"Database sẽ được lưu tại: {KB_PATH}")
    
    # Tạo thư mục chứa database trên DBFS nếu chưa có
    os.makedirs(os.path.dirname(KB_PATH), exist_ok=True)
    
    # Kết nối SQLite
    conn = sqlite3.connect(KB_PATH)

    try:
        build_users_table(conn)
        build_artists_table(conn)
        build_items_table(conn)
        build_stats_table(conn)

        # Tóm tắt
        print("\n── Database Summary ──")
        for table in ["users", "artists", "items", "system_stats"]:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  {table}: {count:,} rows")
            except Exception:
                print(f"  {table}: [không có dữ liệu]")

        db_size = os.path.getsize(KB_PATH) / 1024 / 1024
        print(f"\n  DB size: {db_size:.2f} MB")
        print(f"\n✅ Layer 3 — Knowledge Base hoàn tất!")
        print(f"   → {KB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    run()

# 🎵 Unified Music Recommendation System

Hệ thống gợi ý nhạc tích hợp **4 tầng** — gộp 3 module ML độc lập thành một pipeline hoàn chỉnh.

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                  LAYER 4 — Inference API                        │
│              recommender.py  (LocalRecommender)                 │
│  • Genre-Boost Scoring    • Churn-Guard Reranking               │
│  • MMR Diversity          • Cold-Start Fallback                 │
└───────────────────────┬─────────────────────────────────────────┘
                        │ query SQL
┌───────────────────────▼─────────────────────────────────────────┐
│                  LAYER 3 — Knowledge Base                       │
│              mappings.db  (SQLite)                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  users   │  │ artists  │  │  items   │  │ system_stats  │  │
│  │churn_risk│  │dominant  │  │  msid    │  │ total_users   │  │
│  │persona   │  │genre     │  │  genre   │  │ high_churn    │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ read CSV / NPY
┌───────────────────────▼─────────────────────────────────────────┐
│                  LAYER 2 — Offline ML Engines                   │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐  │
│  │  Engine A      │ │  Engine B      │ │  Engine C          │  │
│  │  02_genre_     │ │  03_churn_     │ │  04_lgcn_          │  │
│  │  persona_      │ │  engine.py     │ │  engine.py         │  │
│  │  engine.py     │ │                │ │                    │  │
│  │ KMeans/GMM+LSH │ │ RandomForest   │ │  LightGCN          │  │
│  │ → user_taste   │ │ → churn_risk   │ │  → user_vectors    │  │
│  │   artist_genre │ │   web_dash_v2  │ │    item_vectors    │  │
│  │   rich_profile │ │                │ │    index_mappings  │  │
│  └────────────────┘ └────────────────┘ └────────────────────┘  │
└───────────────────────┬─────────────────────────────────────────┘
                        │ PySpark clean data
┌───────────────────────▼─────────────────────────────────────────┐
│                  LAYER 1 — Data Lake                            │
│              01_data_pipeline.py                                │
│  • Đọc CSV/XLSX thô → Làm sạch → Chuẩn hóa timestamp          │
│  • Output: data/outputs/clean_data/                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Cấu Trúc Thư Mục

```
unified_music_system/
├── config.py                    # ⚙️  Cấu hình trung tâm
├── requirements.txt             # 📦 Dependencies
├── README.md                    # 📖 Tài liệu này
│
├── 01_data_pipeline.py          # LAYER 1: Làm sạch data (PySpark)
├── 02_genre_persona_engine.py   # LAYER 2A: Genre + Persona (KMeans + LSH)
├── 03_churn_engine.py           # LAYER 2B: Churn Prediction (Random Forest)
├── 04_lgcn_engine.py            # LAYER 2C: LightGCN Training
├── 05_build_knowledge_base.py   # LAYER 3: Build mappings.db (SQLite)
└── recommender.py               # LAYER 4: Inference API
│
data/
├── raw/                         # ← Đặt file CSV/XLSX gốc vào đây
├── outputs/                     # ← Layer 2 tự ghi kết quả vào đây
│   ├── clean_data/
│   ├── user_taste_profile/
│   ├── artist_genre_profile/
│   ├── rich_user_profile/
│   └── web_dashboard_data_v2.csv
├── models/                      # ← Layer 2C ghi embeddings vào đây
│   ├── user_vectors.npy
│   ├── item_vectors.npy
│   ├── index_mappings.pkl
│   └── train_matrix.npz
└── mappings.db                  # ← Layer 3 tổng hợp tất cả vào đây
```

---

## 🚀 Hướng Dẫn Chạy

### Bước 1: Cài đặt
```bash
pip install -r requirements.txt
```

### Bước 2: Đặt dữ liệu
```bash
# Copy file CSV/XLSX log nghe nhạc vào:
cp /your/data/*.csv data/raw/
```

### Bước 3: Chạy tuần tự từng layer
```bash
# Layer 1 — Làm sạch data
python 01_data_pipeline.py

# Layer 2A — Genre + Persona (chạy độc lập, không phụ thuộc 2B/2C)
python 02_genre_persona_engine.py

# Layer 2B — Churn (nên chạy sau 2A để ghép persona)
python 03_churn_engine.py

# Layer 2C — LightGCN (tốn thời gian nhất, ~20 epoch)
python 04_lgcn_engine.py

# Layer 3 — Gom tất cả vào SQLite
python 05_build_knowledge_base.py
```

### Bước 4: Sử dụng Recommender
```python
from recommender import LocalRecommender

rec = LocalRecommender()

# Gợi ý hybrid (tự động genre-boost + churn-guard)
df = rec.recommend_hybrid("user_id_123", n=10)
print(df)

# Xem profile user
profile = rec.get_user_profile("user_id_123")
print(profile)
```

---

## 🎯 Tính Năng Nổi Bật

### 1. Genre-Boost Scoring
```
Nếu bài hát thuộc thể loại yêu thích của user:
  lgcn_score × 1.2  (+20%)
```
→ TF-IDF được làm giàu thêm `dominant_genre` → bài cùng thể loại gần nhau hơn.

### 2. Churn-Guard Reranking
```
Nếu churn_risk > 70%:
  content_alpha = 0      (tắt exploration)
  final_score = 0.4 × lgcn_score + 0.6 × trending_score
```
→ Đẩy nhạc quen thuộc + đang hot để níu chân user.

### 3. Hybrid Scoring Pipeline
```
score = (1 - α) × lgcn_score × genre_boost + α × tfidf_score
```
- `α = 0.25` mặc định
- `α = 0.0`  khi churn cao (tắt TF-IDF exploration)

### 4. MMR Diversity
```
Maximal Marginal Relevance: λ = 0.5
→ Cân bằng relevance và đa dạng thể loại/nghệ sĩ
```

### 5. Multi-mode Recommendations
| Hàm | Mục đích |
|-----|---------|
| `recommend_hybrid` | Chính — Genre+Churn aware |
| `recommend_trending` | Nhạc đang hot |
| `recommend_realtime` | Blend session + long-term |
| `recommend_discovery` | Khám phá ngoài vùng an toàn |
| `recommend_similar_to_new_item` | Cold-item similarity |
| `recommend_similar_users` | Cộng đồng người dùng tương tự |
| `recommend_next_in_session` | Dự đoán bài tiếp theo |
| `recommend_cold_content` | Cold-start user |

---

## ⚙️ Tuỳ Chỉnh

Tất cả tham số quan trọng nằm trong `config.py`:

```python
RECOMMENDER_CONFIG = {
    "default_content_alpha"  : 0.25,   # Trọng số TF-IDF
    "genre_bonus_multiplier" : 1.20,   # +20% genre boost
    "churn_risk_threshold"   : 70.0,   # % ngưỡng bật churn guard
    "churn_trending_weight"  : 0.6,    # Trọng số trending khi churn cao
    "mmr_lambda"             : 0.5,    # Diversity vs relevance
}
```

---

## 📊 Dữ Liệu Cần Thiết

File CSV/XLSX phải có tối thiểu các cột:

| Cột | Mô tả |
|-----|-------|
| `user_id` | ID người dùng |
| `recording_msid` | ID bài hát |
| `track_name` | Tên bài hát |
| `artist_name` | Tên nghệ sĩ |
| `timestamp` | Thời gian nghe (`YYYY-MM-DD HH:MM:SS`) |

Cột bổ sung (tuỳ chọn): `skip`, `duration`

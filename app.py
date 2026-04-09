import streamlit as st
import pandas as pd
import time
import json
from main import LocalRecommender
from openai import OpenAI
from chatbot_improved import render_chatbot_tab
# Cấu hình giao diện toàn trang
st.set_page_config(page_title="Music Recommender AI", page_icon="🎵", layout="wide")

# ==========================================
# 1. TẢI MODEL (Chỉ tải 1 lần duy nhất nhờ Cache)
# ==========================================
@st.cache_resource
def load_model():
    return LocalRecommender(
        model_dir='.',
        db_path='./mappings.db',
        small_pkl='./mappings_small.pkl',
        cache_dir='.'
    )

with st.spinner("Đang tải bộ não AI... (khoảng 30s lần đầu)"):
    rec_sys = load_model()

# ==========================================
# 2. KHỞI TẠO AI CLIENT & CÔNG CỤ (Dùng Ollama Local)
# ==========================================
# Khai báo các công cụ (hàm) cho Trợ lý Ảo

# ==========================================
# 3. XÂY DỰNG GIAO DIỆN (SIDEBAR)
# ==========================================
st.sidebar.title("🎛️ Bảng Điều Khiển")
st.sidebar.markdown("---")

user_input = st.sidebar.text_input("Nhập User ID (VD: 13, 42):", value="13")
n_recs = st.sidebar.slider("Số lượng gợi ý:", min_value=5, max_value=30, value=10, step=5)

feature = st.sidebar.radio(
    "Chọn tính năng Demo:",
    [
        "1. Gợi ý Cá nhân hóa (Hybrid)",
        "2. Top Thịnh hành (Trending)",
        "3. Khám phá mới (Discovery)",
        "4. Mix Playlist với Bài hát mồi",
        "5. Trợ lý Ảo AI (Chatbot)"
    ]
)

# ==========================================
# 4. HIỂN THỊ KẾT QUẢ (MAIN AREA)
# ==========================================
st.title("🎵 Hệ Thống Gợi Ý Âm Nhạc Thông Minh")
st.markdown("Đồ án Khai phá dữ liệu & Hệ khuyến nghị")
st.markdown("---")

if user_input:
    tier, uid = rec_sys._get_user_tier(user_input)
    st.sidebar.info(f"👤 **Khách hàng:** {user_input} | **Trạng thái:** {tier.upper()} User")

    start_time = time.time()
    
    # CÁC TÍNH NĂNG CƠ BẢN
    if feature == "1. Gợi ý Cá nhân hóa (Hybrid)":
        st.subheader("🎯 Dành Riêng Cho Bạn")
        df = rec_sys.recommend_hybrid(user_input, n=n_recs)
        st.dataframe(df, use_container_width=True)

    elif feature == "2. Top Thịnh hành (Trending)":
        st.subheader("🔥 Đang Hot Hiện Nay")
        df = rec_sys.recommend_trending(user_id_str=user_input, n=n_recs, personal_weight=0.5)
        st.dataframe(df, use_container_width=True)

    elif feature == "3. Khám phá mới (Discovery)":
        st.subheader("🚀 Thoát Khỏi Vùng An Toàn")
        df = rec_sys.recommend_discovery(user_input, n=n_recs, serendipity=0.4)
        st.dataframe(df, use_container_width=True)

    elif feature == "4. Mix Playlist với Bài hát mồi":
        st.subheader("🎧 Tạo Playlist Mix")
        seed_song = st.text_input("Nhập tên bài hát bạn đang nghiện (VD: Creep):", value="")
        if st.button("Tạo Playlist Ngay") and seed_song:
            df = rec_sys.generate_playlist(user_input, seed_track_names=[seed_song], n_songs=n_recs)
            st.dataframe(df, use_container_width=True)

    # TÍNH NĂNG TRỢ LÝ ẢO AI
    # TÍNH NĂNG TRỢ LÝ ẢO AI
    
    elif feature == "5. Trợ lý Ảo AI (Chatbot)":
        render_chatbot_tab(rec_sys, user_input, n_recs)  # chỉ cần 1 dòng này
    # Hiển thị tốc độ phản hồi chung cho các tab 1, 2, 3, 4 (ẩn ở tab 5 vì đã đo riêng)
    if feature != "5. Trợ lý Ảo AI (Chatbot)":
        st.caption(f"⚡ Thời gian phản hồi: {time.time() - start_time:.4f} giây")
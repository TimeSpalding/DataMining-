import streamlit as st
import pandas as pd
import time
from datetime import datetime
from src.core.recommender import LocalRecommender

# --- IMPORT CÁC MODULE GIAO DIỆN ĐÃ TÁCH ---
from src.ui.tab_home import render_home_tab
from src.ui.tab_discovery import render_discovery_tab
from src.ui.chatbot import render_chatbot_tab
from src.ui.tab_context import render_context_tab
from src.ui.admin_dashboard import render_admin_dashboard
from src.ui.components import inject_custom_css, render_bottom_player

# ==========================================
# 0. CHUẨN BỊ (SESSION STATE & CONFIG)
# ==========================================
st.set_page_config(
    page_title="Music Recommender AI", 
    page_icon="https://cdn-icons-png.flaticon.com/512/3844/3844724.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Khởi tạo session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "role" not in st.session_state:
    st.session_state.role = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "Guest"

# Kích hoạt CSS giao diện Spotify
inject_custom_css()

# ==========================================
# 1. TRANG ĐĂNG NHẬP (LOGIN PAGE)
# ==========================================
def render_login_page():
    # Căn giữa trang login
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3844/3844724.png", width=100)
        st.title("Chào mừng đến với Music AI")
        st.markdown("Vui lòng đăng nhập để tiếp tục trải nghiệm cá nhân hóa.")
        
        with st.container(border=True):
            user_id = st.text_input("ID Người dùng", placeholder="Nhập ID (VD: admin, user, 13...)")
            password = st.text_input("Mật khẩu", type="password", placeholder="Mặc định: 123456789")
            
            login_btn = st.button("Đăng nhập", use_container_width=True, type="primary")
            st.divider()
            guest_btn = st.button("Trải nghiệm nhanh (Cold User)", use_container_width=True)

        if login_btn:
            if password == "123456789":
                st.session_state.authenticated = True
                if user_id.lower() == "admin":
                    st.session_state.role = "admin"
                    st.session_state.user_id = "Admin"
                else:
                    st.session_state.role = "user"
                    st.session_state.user_id = user_id if user_id else "13"
                st.rerun()
            else:
                st.error("Sai mật khẩu! Vui lòng thử lại.")
        
        if guest_btn:
            st.session_state.authenticated = True
            st.session_state.role = "guest"
            st.session_state.user_id = "Guest"
            st.rerun()

# ==========================================
# 2. KIỂM TRA QUYỀN TRUY CẬP
# ==========================================
if not st.session_state.authenticated:
    render_login_page()
    st.stop()

# ==========================================
# 3. TẢI MODEL (Sau khi đã đăng nhập)
# ==========================================
@st.cache_resource
def load_model():
    return LocalRecommender(
        model_dir='./model',
        db_path='./model/mappings.db',
        small_pkl='./model/mappings_small.pkl',
        cache_dir='./model'
    )

with st.spinner("Đang khởi động bộ não AI..."):
    rec_sys = load_model()

# ==========================================
# 4. SIDEBAR - ĐIỀU KHIỂN CHÍNH
# ==========================================
st.sidebar.title("Bảng Điều Khiển")
st.sidebar.markdown(f"Chào mừng, **{st.session_state.user_id}** ({st.session_state.role.upper()})")

if st.sidebar.button("Đăng xuất"):
    st.session_state.authenticated = False
    st.session_state.role = None
    st.rerun()

st.sidebar.markdown("---")

# Tùy chọn menu dựa trên vai trò
menu_options = []
if st.session_state.role == "admin":
    menu_options = [
        "Quản Trị Rời Bỏ (Churn)",
        "Trang Chủ & Cá Nhân",
        "Khám Phá & Xu Hướng",
        "Trợ Lý Ảo AI"
    ]
elif st.session_state.role == "user":
    menu_options = [
        "Trang Chủ & Cá Nhân",
        "Khám Phá & Xu Hướng",
        "Playlist Của Bạn",
        "Trợ Lý Ảo AI"
    ]
else: # guest
    menu_options = [
        "Khám Phá & Xu Hướng",
        "Trợ Lý Ảo AI"
    ]

category = st.sidebar.selectbox("Chọn Nhóm Trải Nghiệm:", menu_options)

# Cấu hình tham số cho Recommender
n_recs = 10
if st.session_state.role != "guest":
    n_recs = st.sidebar.slider("Số lượng gợi ý:", min_value=5, max_value=30, value=10, step=5)

# ==========================================
# 5. NỘI DUNG CHÍNH (MAIN AREA)
# ==========================================
st.title("Hệ Thống Gợi Ý Âm Nhạc Thông Minh")
st.markdown("---")

start_time = time.time()
user_input = st.session_state.user_id if st.session_state.role == "user" else "13"

if category == "Quản Trị Rời Bỏ (Churn)":
    render_admin_dashboard()

elif category == "Trang Chủ & Cá Nhân":
    render_home_tab(rec_sys, user_input, n_recs)

elif category == "Khám Phá & Xu Hướng":
    render_discovery_tab(rec_sys, user_input, n_recs)

elif category == "Playlist Của Bạn":
    render_context_tab(rec_sys, user_input, n_recs)

elif category == "Trợ Lý Ảo AI":
    render_chatbot_tab(rec_sys, user_input, n_recs)

# ==========================================
# 6. FOOTER
# ==========================================
if category not in ["Trợ Lý Ảo AI", "Quản Trị Rời Bỏ (Churn)"]:
    st.markdown("---")
    st.caption(f"Phản hồi trong: {time.time() - start_time:.4f} giây")
    render_bottom_player()
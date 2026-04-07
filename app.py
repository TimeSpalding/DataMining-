<<<<<<< HEAD
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Music Churn Dashboard", page_icon="🎶", layout="wide")
st.title("🎶 Hệ Thống Quản Trị Rời Bỏ & Giữ Chân Người Dùng")
st.markdown("Dashboard hỗ trợ Giám đốc Sản phẩm ra quyết định dựa trên dữ liệu (Data-Driven Decision Making).")

# --- 2. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ---
@st.cache_data
def load_data():
    df = pd.read_csv("web_dashboard_data_v2.csv")
    # Tạo thêm nhãn Phân loại rủi ro cho biểu đồ tổng quan
    def categorize_risk(score):
        if score > 70: return "🚨 Rủi ro cao (>70%)"
        elif score > 40: return "⚠️ Cảnh báo (40-70%)"
        else: return "✅ An toàn (<40%)"
    
    df['Risk_Level'] = df['churn_risk_percent'].apply(categorize_risk)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'web_dashboard_data_v2.csv'. Vui lòng kiểm tra lại đường dẫn!")
    st.stop()

# --- 3. CHIA GIAO DIỆN THÀNH 2 TAB ---
tab1, tab2 = st.tabs(["📊 TỔNG QUAN HỆ THỐNG (MACRO)", "👤 PHÂN TÍCH CÁ NHÂN & KÊ ĐƠN (MICRO)"])

# ==========================================
# TAB 1: GÓC NHÌN QUẢN TRỊ VIÊN (MACRO)
# ==========================================
with tab1:
    st.header("1. Sức khỏe tổng thể của Nền tảng")
    
    # KPIs Tổng quan
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Tổng số User đang quản lý", f"{len(df):,} người")
    kpi_col2.metric("Nguy cơ Churn Trung bình", f"{df['churn_risk_percent'].mean():.2f}%")
    high_risk_count = len(df[df['churn_risk_percent'] > 70])
    kpi_col3.metric("Số User sắp rời bỏ (Cần cứu)", f"{high_risk_count} người", delta="Báo động đỏ", delta_color="inverse")
    
    st.divider()
    st.header("2. Phân tích Phân bổ & Tương quan")
    
    col_pie, col_scatter = st.columns([1, 2])
    
    # Biểu đồ tròn: Tỷ lệ phân bổ rủi ro
    with col_pie:
        st.subheader("Phân bổ Tập người dùng")
        pie_data = df['Risk_Level'].value_counts().reset_index()
        pie_data.columns = ['Mức độ', 'Số lượng']
        fig_pie = px.pie(pie_data, values='Số lượng', names='Mức độ', 
                         color='Mức độ',
                         color_discrete_map={
                             "✅ An toàn (<40%)": "lightgreen",
                             "⚠️ Cảnh báo (40-70%)": "orange",
                             "🚨 Rủi ro cao (>70%)": "red"
                         }, hole=0.4)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Biểu đồ phân tán: Tìm ra quy luật những ai hay bỏ app
    with col_scatter:
        st.subheader("Tương quan: Tuổi đời tài khoản vs Rủi ro rời bỏ")
        # Phân tích xem có phải user mới dễ bỏ đi nhất không?
        fig_scatter = px.scatter(df, x="tenure_days", y="churn_risk_percent", 
                                 color="Risk_Level", 
                                 size="total_listens", # Chấm to nhỏ theo độ ghiền
                                 hover_data=['user_id', 'daily_listen_rate'],
                                 color_discrete_map={
                                     "✅ An toàn (<40%)": "lightgreen",
                                     "⚠️ Cảnh báo (40-70%)": "orange",
                                     "🚨 Rủi ro cao (>70%)": "red"
                                 },
                                 labels={"tenure_days": "Số ngày gắn bó (Tenure)", "churn_risk_percent": "Xác suất Rời bỏ (%)"})
        # Vẽ thêm đường cảnh báo 70%
        fig_scatter.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Ngưỡng Báo Động")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 2: ĐÀO SÂU CÁ NHÂN & KÊ ĐƠN (MICRO)
# ==========================================
with tab2:
    st.sidebar.header("🔍 Tra cứu Người dùng")
    user_ids = df['user_id'].astype(str).tolist()
    selected_user = st.sidebar.selectbox("Nhập hoặc chọn ID Người dùng:", ["-- Chọn User --"] + user_ids)

    if selected_user != "-- Chọn User --":
        user_data = df[df['user_id'].astype(str) == selected_user].iloc[0]
        
        st.markdown(f"### 👤 Hồ sơ chi tiết User ID: `{selected_user}`")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            risk_score = user_data['churn_risk_percent']
            if risk_score > 70:
                risk_color = "red"
                status_text = "🚨 NGUY CƠ CAO"
            elif risk_score > 40:
                risk_color = "orange"
                status_text = "⚠️ THEO DÕI THÊM"
            else:
                risk_color = "green"
                status_text = "✅ AN TOÀN"

            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk_score, title = {'text': "Xác suất Rời bỏ (%)"},
                gauge = {
                    'axis': {'range': [None, 100]}, 'bar': {'color': risk_color},
                    'steps': [{'range': [0, 40], 'color': "lightgreen"},
                              {'range': [40, 70], 'color': "navajowhite"},
                              {'range': [70, 100], 'color': "lightpink"}],
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(f"**Trạng thái:** {status_text}")
            
            # --- TÍNH NĂNG "ĂN TIỀN": PRESCRIPTIVE ANALYTICS ---
            st.divider()
            st.subheader("💡 Đề xuất Hành động (Auto-Trigger)")
            if risk_score > 70:
                st.error("**Kích hoạt Kịch bản Giữ chân Khẩn cấp!**")
                if user_data['artist_diversity'] < 0.3:
                    st.write("👉 **Nguyên nhân dự đoán:** User là 'Fan cứng' nhưng có thể đã cày hết nhạc của Idol.")
                    st.write("🛠️ **Hành động (Gửi tới Module 3):** Gửi Push Notification giới thiệu Playlist 'Nghệ sĩ tương đồng' để mở rộng gu nghe nhạc.")
                else:
                    st.write("👉 **Nguyên nhân dự đoán:** User thích khám phá nhưng hệ thống đang bí ý tưởng.")
                    st.write("🛠️ **Hành động (Gửi tới Module 3):** Tặng mã VIP 1 tuần và đẩy mạnh thuật toán LSI (Module 2) gợi ý các bài hát Indie mới nhất.")
            elif risk_score > 40:
                st.warning("**Kích hoạt Kịch bản Hâm nóng:**")
                st.write("🛠️ **Hành động:** Gửi Email tổng kết 'Nhìn lại 1 tháng nghe nhạc của bạn' để tăng tương tác.")
            else:
                st.success("**Người dùng trung thành:**")
                st.write("🛠️ **Hành động:** Không can thiệp, tiếp tục duy trì thuật toán Khuyến nghị hiện tại.")

        with col2:
            st.subheader("📊 Các chỉ số Hành vi")
            k1, k2, k3 = st.columns(3)
            k1.metric(label="Tổng số bài đã nghe", value=f"{int(user_data['total_listens'])}")
            k2.metric(label="Tuổi đời (Ngày)", value=f"{int(user_data['tenure_days'])}")
            k3.metric(label="Tốc độ (Bài/Ngày)", value=f"{user_data['daily_listen_rate']}")
            
            st.write("") 
            
            habit_df = pd.DataFrame({
                'Chỉ số': ['Tỷ lệ nghe Đêm (Cảm xúc)', 'Độ đa dạng Ca sĩ (Khám phá)'],
                'Giá trị (%)': [user_data['night_listen_ratio'] * 100, user_data['artist_diversity'] * 100]
            })
            
            fig_bar = px.bar(habit_df, x='Giá trị (%)', y='Chỉ số', orientation='h', 
                             title="Phân tích Đặc điểm Tâm lý", range_x=[0, 100], text='Giá trị (%)', color='Chỉ số')
            fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside', showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.info("👈 Vui lòng chọn một User ID ở thanh bên trái để xem phân tích chi tiết.")
=======
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import scipy.sparse as sp
import os

# 1. Cấu hình trang Web
st.set_page_config(
    page_title="Hệ Khuyến Nghị Âm Nhạc", 
    page_icon="🎵", 
    layout="wide"
)

# 2. Load Model (Dùng @st.cache_resource để lưu model vào RAM, không load lại liên tục)
@st.cache_resource
def load_recommender(model_dir='model'):
    class MusicRecommender:
        def __init__(self, model_dir):
            self.model     = joblib.load(os.path.join(model_dir, 'als_model.pkl'))
            mappings       = joblib.load(os.path.join(model_dir, 'index_mappings.pkl'))
            self.user2idx  = mappings['user2idx']
            self.item2idx  = mappings['item2idx']
            self.idx2user  = mappings['idx2user']
            self.idx2item  = mappings['idx2item']
            self.user_item = sp.load_npz(os.path.join(model_dir, 'user_item_matrix.npz'))
            # Đọc parquet để lấy metadata
            self.item_meta = pd.read_parquet(os.path.join(model_dir, 'item_metadata.parquet'))

        def _item_idx_to_df(self, item_ids, scores):
            rows = []
            for idx, score in zip(item_ids, scores):
                key = self.idx2item[idx]
                track, artist = key.split('|||')
                rows.append({
                    'Bài hát': track, 
                    'Nghệ sĩ': artist, 
                    'Độ tin cậy': round(float(score), 4)
                })
            return pd.DataFrame(rows)

        def user_history(self, user_id, n=10):
            uid_str = str(user_id)
            if uid_str not in self.user2idx: 
                return pd.DataFrame()
            uid = self.user2idx[uid_str]
            row = self.user_item[uid]
            top_n = np.argsort(row.data)[::-1][:n]
            df = self._item_idx_to_df(row.indices[top_n], row.data[top_n])
            df.rename(columns={'Độ tin cậy':'Trọng số tương tác'}, inplace=True)
            df.index = range(1, len(df)+1)
            return df

        def recommend_for_user(self, user_id, n=10, filter_listened=True):
            uid_str = str(user_id)
            if uid_str not in self.user2idx:
                return self.popular_items(n)
            uid = self.user2idx[uid_str]
            rec_ids, scores = self.model.recommend(
                uid, self.user_item[uid], N=n,
                filter_already_liked_items=filter_listened
            )
            df = self._item_idx_to_df(list(rec_ids), list(scores))
            df.index = range(1, len(df)+1)
            return df

        def popular_items(self, n=10):
            item_counts = np.asarray(self.user_item.sum(axis=0)).flatten()
            top_ids = np.argsort(item_counts)[::-1][:n]
            df = self._item_idx_to_df(top_ids, item_counts[top_ids])
            df.rename(columns={'Độ tin cậy':'Tổng lượt nghe'}, inplace=True)
            df.index = range(1, len(df)+1)
            return df

    return MusicRecommender(model_dir)

# Bắt lỗi nếu thiếu thư mục/file model
try:
    rec = load_recommender('model')
except Exception as e:
    st.error(f"⚠️ Lỗi khởi tạo mô hình. Hãy kiểm tra lại thư mục 'model'. Chi tiết: {e}")
    st.stop()

# 3. Xây dựng Giao diện UI
st.title("🎵 Hệ Thống Khuyến Nghị Âm Nhạc Cá Nhân Hóa")
st.markdown("---")

# Thanh điều hướng (Sidebar)
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    user_id_input = st.text_input("👤 Nhập ID Người Dùng (VD: 1, 5, 10):", value="1")
    top_n = st.slider("📊 Số lượng bài hát gợi ý:", min_value=5, max_value=50, value=10)
    filter_listened = st.checkbox("🚫 Ẩn các bài đã nghe", value=True)
    
    st.markdown("---")
    st.caption(f"Trạng thái Model: Sẵn sàng\n\nTổng Users: {len(rec.idx2user):,}\n\nTổng Items: {len(rec.idx2item):,}")

# Khu vực hiển thị chính
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎧 Lịch sử nghe nhạc")
    if user_id_input:
        history_df = rec.user_history(user_id_input, n=top_n)
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.warning("Người dùng này chưa có lịch sử nghe nhạc trong hệ thống.")

with col2:
    st.subheader("✨ Gợi ý dành riêng cho User")
    if user_id_input:
        recommendations_df = rec.recommend_for_user(
            user_id_input, 
            n=top_n, 
            filter_listened=filter_listened
        )
        
        # Nhận diện Cold-start nếu trả về cột 'Tổng lượt nghe' thay vì 'Độ tin cậy'
        if 'Tổng lượt nghe' in recommendations_df.columns:
            st.info("User mới (Cold-start). Dưới đây là các bài hát thịnh hành nhất:")
        else:
            st.success("Mô hình ALS đã lọc và gợi ý thành công!")
            
        st.dataframe(recommendations_df, use_container_width=True)
>>>>>>> 39c3eaa9f544f3bcc3a04ff371c85cfb19470cb0

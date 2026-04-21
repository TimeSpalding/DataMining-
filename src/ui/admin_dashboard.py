import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.core.churn_processor import load_churn_data

def render_admin_dashboard():
    st.title("🎶 Hệ Thống Quản Trị Rời Bỏ & Giữ Chân Người Dùng")
    st.markdown("Dashboard hỗ trợ Giám đốc Sản phẩm ra quyết định dựa trên dữ liệu (Data-Driven Decision Making).")

    # --- 2. ĐỌC VÀ CHUẨN BỊ DỮ LIỆU ---
    try:
        df = load_churn_data()
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {e}")
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
            fig_scatter = px.scatter(df, x="tenure_days", y="churn_risk_percent", 
                                     color="Risk_Level", 
                                     size="total_listens", 
                                     hover_data=['user_id', 'daily_listen_rate'],
                                     color_discrete_map={
                                         "✅ An toàn (<40%)": "lightgreen",
                                         "⚠️ Cảnh báo (40-70%)": "orange",
                                         "🚨 Rủi ro cao (>70%)": "red"
                                     },
                                     labels={"tenure_days": "Số ngày gắn bó (Tenure)", "churn_risk_percent": "Xác suất Rời bỏ (%)"})
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
                
                st.divider()
                st.subheader("💡 Đề xuất Hành động (Auto-Trigger)")
                if risk_score > 70:
                    st.error("**Kích hoạt Kịch bản Giữ chân Khẩn cấp!**")
                    if user_data['artist_diversity'] < 0.3:
                        st.write("👉 **Nguyên nhân dự đoán:** User là 'Fan cứng' nhưng có thể đã cày hết nhạc của Idol.")
                        st.write("🛠️ **Hành động:** Gửi Push Notification giới thiệu Playlist 'Nghệ sĩ tương đồng'.")
                    else:
                        st.write("👉 **Nguyên nhân dự đoán:** User thích khám phá nhưng hệ thống đang bí ý tưởng.")
                        st.write("🛠️ **Hành động:** Tặng mã VIP 1 tuần và đẩy mạnh thuật toán LSI gợi ý các bài hát Indie mới nhất.")
                elif risk_score > 40:
                    st.warning("**Kích hoạt Kịch bản Hâm nóng:**")
                    st.write("🛠️ **Hành động:** Gửi Email tổng kết 'Nhìn lại 1 tháng nghe nhạc của bạn'.")
                else:
                    st.success("**Người dùng trung thành:**")
                    st.write("🛠️ **Hành động:** Không can thiệp, tiếp tục duy trì thuật toán hiện tại.")

            with col2:
                st.subheader("📊 Các chỉ số Hành vi")
                k1, k2, k3 = st.columns(3)
                k1.metric(label="Tổng số bài đã nghe", value=f"{int(user_data['total_listens'])}")
                k2.metric(label="Tuổi đời (Ngày)", value=f"{int(user_data['tenure_days'])}")
                k3.metric(label="Tốc độ (Bài/Ngày)", value=f"{user_data['daily_listen_rate']}")
                
                habit_df = pd.DataFrame({
                    'Chỉ số': ['Tỷ lệ nghe Đêm (%)', 'Độ đa dạng Ca sĩ (%)'],
                    'Giá trị (%)': [user_data['night_listen_ratio'] * 100, user_data['artist_diversity'] * 100]
                })
                
                fig_bar = px.bar(habit_df, x='Giá trị (%)', y='Chỉ số', orientation='h', 
                                 title="Phân tích Đặc điểm Tâm lý", range_x=[0, 100], text='Giá trị (%)', color='Chỉ số')
                fig_bar.update_traces(texttemplate='%{text:.1f}%', textposition='outside', showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        else:
            st.info("👈 Vui lòng chọn một User ID ở thanh bên trái để xem phân tích chi tiết.")

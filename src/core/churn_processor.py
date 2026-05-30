import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_churn_data(file_path=None):
    """
    Đọc dữ liệu từ file CSV và thực hiện tiền xử lý nhãn rủi ro.
    """
    if file_path is None:
        # Mặc định tìm trong thư mục data của project
        file_path = os.path.join("data", "web_dashboard_data_v2.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {file_path}")

    df = pd.read_csv(file_path)
    
    # Tạo thêm nhãn Phân loại rủi ro cho biểu đồ tổng quan
    def categorize_risk(score):
        if score > 70: return "🚨 Rủi ro cao (>70%)"
        elif score > 40: return "⚠️ Cảnh báo (40-70%)"
        else: return "✅ An toàn (<40%)"
    
    df['Risk_Level'] = df['churn_risk_percent'].apply(categorize_risk)
    return df

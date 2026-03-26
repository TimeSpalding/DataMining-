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
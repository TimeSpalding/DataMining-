import glob
import json
import pandas as pd
import streamlit as st
from azure.eventhub import EventHubProducerClient, EventData

# --- 1. CẤU HÌNH KẾT NỐI AZURE (Lưu Cache để không bị kết nối lại nhiều lần) ---
CONNECTION_STRING = os.getenv("AZURE_CONNECTION_STRING")

@st.cache_resource
def get_producer_client():
    return EventHubProducerClient.from_connection_string(
        conn_str=CONNECTION_STRING, 
        eventhub_name=EVENT_HUB_NAME
    )

producer = get_producer_client()

# --- 2. TẢI DỮ LIỆU TỪ CSV ---
@st.cache_data
def load_data():
    folder_path = "E:/Ky2_2025_2026/DataMining-/CLEARDATA/*.csv" # Đổi lại đường dẫn đúng của bạn
    file_list = glob.glob(folder_path)
    if not file_list:
        st.error("❌ Không tìm thấy file CSV nào!")
        return pd.DataFrame()
    
    # Load 5000 dòng đầu tiên làm kho đạn
    return pd.read_csv(file_list[0], nrows=5000)

df = load_data()

# --- 3. QUẢN LÝ TRẠNG THÁI (Ghi nhớ đang phát tới bài nhạc nào) ---
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# --- 4. XÂY DỰNG GIAO DIỆN WEB UI ---
st.set_page_config(page_title="Music Streaming Simulator", page_icon="🎵")

st.title("🎧 Music App: Streaming Simulator")
st.markdown("Giả lập hành vi người dùng nghe nhạc Real-time để bắn lên **Azure Event Hubs**.")

# Khung hiển thị trạng thái
st.info(f"Kho dữ liệu: Đã tải {len(df)} sự kiện. Đang chờ kích hoạt...")

# Layout chia cột cho đẹp
col1, col2 = st.columns([1, 2])

with col1:
    # Nút bấm ma thuật
    play_button = st.button("▶️ PLAY NHẠC (Bắn 1 Event)", type="primary", use_container_width=True)

with col2:
    if play_button:
        if df.empty:
            st.warning("Dữ liệu trống, không thể phát!")
        elif st.session_state.current_index >= len(df):
            st.warning("Đã phát hết danh sách nhạc trong CSV!")
        else:
            # Lấy dòng dữ liệu hiện tại dựa vào index
            row = df.iloc[st.session_state.current_index]
            
            # Đóng gói
            event_dict = {
                "user_id": str(row['user_id']),
                "timestamp": str(row['timestamp']),
                "recording_msid": str(row['recording_msid'])
            }
            event_json = json.dumps(event_dict)
            
            # Bắn lên Azure
            try:
                event_data = EventData(event_json)
                producer.send_batch([event_data])
                
                # Hiển thị thông báo thành công màu xanh lá
                st.success(f"✅ Đã gửi tín hiệu lên Azure: User **{event_dict['user_id']}** vừa play bài hát **{event_dict['recording_msid']}** lúc {event_dict['timestamp']}")
                
                # Tăng biến đếm để lần click sau sẽ lấy bài nhạc tiếp theo
                st.session_state.current_index += 1
                
            except Exception as e:
                st.error(f"❌ Lỗi truyền tải: {e}")

# Hiển thị log nhỏ ở dưới cùng
st.markdown("---")
st.caption(f"Sự kiện thứ: {st.session_state.current_index} / {len(df)}")
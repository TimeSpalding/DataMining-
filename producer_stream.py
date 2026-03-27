import glob
import time
import json
import pandas as pd
from azure.eventhub import EventHubProducerClient, EventData

print("--- KHỞI ĐỘNG TRẠM PHÁT DỮ LIỆU LUỒNG (PRODUCER) ---")

CONNECTION_STRING = "Endpoint=sb://YOUR_NAMESPACE.servicebus.windows.net/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=YOUR_SECRET_KEY_HERE"
EVENT_HUB_NAME = "music-log-stream"

producer = EventHubProducerClient.from_connection_string(
    conn_str=CONNECTION_STRING, 
    eventhub_name=EVENT_HUB_NAME
)

folder_path = "E:/Ky2_2025_2026/DataMining-/CLEARDATA/*.csv"
file_list = glob.glob(folder_path)

if len(file_list) == 0:
    print("❌ Không tìm thấy file CSV nào trong thư mục!")
    exit()

target_file = file_list[0]
print(f"Đang nạp đạn (dữ liệu) từ file: {target_file}")

df = pd.read_csv(target_file, nrows=5000)

print(f"Sẵn sàng stream {len(df)} sự kiện lên Azure Event Hubs...")

try:
    with producer:
        for index, row in df.iterrows():
            event_dict = {
                "user_id": str(row['user_id']),
                "timestamp": str(row['timestamp']),
                "recording_msid": str(row['recording_msid'])
            }
            
            event_json = json.dumps(event_dict)
            event_data = EventData(event_json)
            
            producer.send_batch([event_data])
            
            print(f"[Đã gửi Real-time] User {event_dict['user_id']} nghe nhạc lúc {event_dict['timestamp']}")
            
            time.sleep(0.5)
            
except KeyboardInterrupt:
    print("\n⏹️ Đã dừng luồng phát dữ liệu chủ động (Ctrl+C)!")
except Exception as e:
    print(f"\n❌ Lỗi truyền tải: {e}")

print("--- KẾT THÚC PHIÊN PHÁT DỮ LIỆU ---")
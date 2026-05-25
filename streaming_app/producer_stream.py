import time
import json
import uuid
import random
from datetime import datetime, timezone
from azure.storage.blob import BlobServiceClient

print("--- KHỞI ĐỘNG TRẠM PHÁT DỮ LIỆU LUỒNG (SINH DỮ LIỆU TỰ ĐỘNG) ---")

CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=musicprojectdm;AccountKey=FR9y9JaINWhWWrSAdOu4FFlaJc3RedM6P9CDHgB2YemEYdzal6I62O3DmZyEGrjUIPkWp7FcBwl4+AStZ0hmwg==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "bronzelive"  

blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

ARTISTS = ["Taylor Swift", "The Weeknd", "Sơn Tùng M-TP", "BlackPink", "Ed Sheeran", "HIEUTHUHAI"]
TRACKS = ["Cruel Summer", "Blinding Lights", "Chúng Ta Của Tương Lai", "How You Like That", "Shape of You", "Ngủ Một Mình"]
RELEASES = ["Lover", "After Hours", "Single 2024", "The Album", "Divide", "Ai Cũng Phải Bắt Đầu Từ Đâu Đó"]

print("Sẵn sàng sinh dữ liệu Live vô hạn...")

try:
    while True:
        random_index = random.randint(0, len(ARTISTS) - 1)
        
        event_dict = {
            "user_id": f"USER_{random.randint(1000, 9999)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artist_name": ARTISTS[random_index],
            "recording_msid": str(uuid.uuid4()),
            "track_name": TRACKS[random_index],
            "release_name": RELEASES[random_index]
        }
        
        event_json = json.dumps(event_dict)
        file_name = f"log_{uuid.uuid4()}.json" 
        
        blob_client = container_client.get_blob_client(file_name)
        blob_client.upload_blob(event_json, overwrite=True)
        
        print(f"[{event_dict['timestamp'][:19]}] Bắn Live 🚀 User {event_dict['user_id']} đang nghe: {event_dict['track_name']}")
        time.sleep(random.uniform(0.5, 2.0))

except KeyboardInterrupt:
    print("\n⏹ Đã dừng luồng phát dữ liệu chủ động (Ctrl+C)!")
except Exception as e:
    print(f"\n❌ Lỗi truyền tải: {e}")
# Music User Recommendation & Analytics

Dự án này tập trung vào việc khai phá dữ liệu nghe nhạc (từ ListenBrain) để thấu hiểu hành vi người dùng, dự báo rời bỏ và cung cấp các gợi ý cá nhân hóa chính xác nhất.

## Cấu trúc hệ thống

Hệ thống được chia thành các thành phần chính:

### 1. Các Module Phân tích & Học máy (Core ML & Pipelines)
* **Module 1: Phân cụm (Clustering)**: Phân loại User dựa trên hành vi nghe nhạc. Xác định các chân dung khách hàng như "Cú đêm", "Fan cứng", hay "Người mới". (Nằm trong thư mục `modules/ml_churn_prediction/01_eda_and_clustering.py`).
* **Module 2: Phân loại Thể loại**: Xác định thể loại âm nhạc của nghệ sĩ cũng như thể loại âm nhạc mà người dùng hay nghe dựa trên dữ liệu nghe nhạc.
* **Module 3: Hệ khuyến nghị (RecSys)**: Triển khai các thuật toán gợi ý như Collaborative Filtering (ALS) và mô hình đồ thị nâng cao **LightGCN** kết hợp với **TF-IDF**.
* **Module 4: Churn Prediction**: Dự báo tỷ lệ người dùng rời bỏ ứng dụng (Churn Prediction).

### 2. Giao diện Người dùng Web Dashboard (Streamlit UI App)
Giao diện trực quan tích hợp toàn bộ kết quả phân tích và công cụ gợi ý thông minh thời gian thực:
* **Dành riêng cho bạn**: Gợi ý cá nhân hóa dựa trên hành vi nghe nhạc lịch sử (sử dụng Hybrid model).
* **Khám phá & Xu hướng**: Cập nhật các bài hát đang thịnh hành và đề xuất mới mẻ thoát khỏi vùng an toàn.
* **Playlist Generator**: Tự động tạo danh sách phát dựa trên bài hát hoặc nghệ sĩ yêu thích.
* **AI Chatbot**: Trò chuyện và yêu cầu gợi ý âm nhạc thông qua trợ lý AI (tích hợp API).
* **Quản trị Rời bỏ (Churn)**: Dashboard trực quan dành cho quản trị viên theo dõi tỷ lệ người dùng rời bỏ (Churn Risk) và đưa ra các đề xuất giữ chân tự động.
* **Trình phát nhạc tích hợp**: Nghe nhạc trực tiếp trên trình duyệt với giao diện hiện đại.

---

## Cấu trúc thư mục

```text
.
├── app.py                      # Giao diện chính Streamlit (Entry point)
├── requirements.txt            # Thư viện phụ thuộc cho UI & ML
├── Dockerfile                  # Cấu hình container hóa ứng dụng
├── src/                        # Mã nguồn của ứng dụng Streamlit UI
│   ├── core/                   # Các engine điều khiển & gợi ý (LocalRecommender)
│   └── ui/                     # Giao diện người dùng (Dashboard, Chatbot, Home, Discovery...)
├── modules/                    # Các module huấn luyện mô hình & phân tích học máy
│   ├── ml_churn_prediction/    # EDA, Clustering và huấn luyện mô hình Churn
│   └── ml_recommendation/      # Huấn luyện mô hình LightGCN và serving online
├── feature_Engineering/        # Pipeline xử lý dữ liệu thô (Bronze, Silver, Gold)
├── streaming_app/              # Ứng dụng mô phỏng luồng dữ liệu nghe nhạc
├── notebooks/                  # File Jupyter Notebook thử nghiệm mô hình
├── data/                       # Chứa dữ liệu (SQLite DB, file nhạc mp3 chạy demo...)
└── assets/                     # Hình ảnh minh họa cho tài liệu
```

---

## Hướng dẫn cài đặt & Sử dụng

### 1. Yêu cầu hệ thống
* Python 3.9+
* RAM: Tối thiểu 8GB (Hệ thống đã được tối ưu hóa sử dụng SQLite Proxy để tiết kiệm RAM).

### 2. Chuẩn bị dữ liệu
> [!IMPORTANT]
> Do dung lượng lớn, thư mục `model/` và `data/songs/` không được đưa lên Git.
> - **Model**: Tải các tệp model tại [Kaggle Dataset](https://www.kaggle.com/datasets/b22dckh068donngkhoa/lightgcn-model). Giải nén và đặt tất cả vào thư mục `model/`.
> - **Âm nhạc**: Thêm các file nhạc dạng `.mp3` vào thư mục `data/songs/` để tính năng Player hoạt động.

### 3. Cài đặt thư viện
Mở terminal tại thư mục gốc và chạy lệnh:
```bash
pip install -r requirements.txt
```

### 4. Khởi chạy ứng dụng Streamlit Dashboard
Chạy lệnh sau để bắt đầu trải nghiệm:
```bash
streamlit run app.py
```

---

## Triển khai với Docker

Nếu bạn muốn chạy ứng dụng trong container để đảm bảo tính nhất quán:

### 1. Build Image
```bash
docker build -t music-recommender .
```

### 2. Run Container
```bash
docker run -p 8501:8501 music-recommender
```
Sau đó truy cập: `http://localhost:8501`

---

## Giao diện Demo

Dưới đây là một số hình ảnh thực tế từ hệ thống:

### 1. Giao diện Trang chủ & Gợi ý Hybrid
![Giao diện Trang chủ](./assets/demo_hybrid.png)

### 2. Trợ lý ảo AI Chatbot
![Giao diện Chatbot](./assets/demo_chatbot.png)

---

Link phân công công việc: https://docs.google.com/spreadsheets/d/11epef7jHn0fEMzOTHz0TlGQOkU2Z5Lff1clV7m3yo8g/edit?gid=0#gid=0
Báo cáo: https://github.com/TimeSpalding/DataMining-/blob/nhanh-cua-khoa/B%C3%A1o%20c%C3%A1o.pdf

# Hệ thống Gợi ý Hybrid (LightGCN + TF-IDF) - Phiên bản Local 8GB RAM

Hệ thống gợi ý âm nhạc tiên tiến kết hợp giữa **Collaborative Filtering** (thuật toán đồ thị LightGCN) và **Content-based Filtering** (phân tích văn bản TF-IDF/SVD). Phiên bản này được thiết kế để giải quyết bài toán xử lý 11.2 triệu bài hát trên máy tính cá nhân có cấu hình RAM hạn chế (8GB).

## 1. Chuẩn bị Dữ liệu
Để hệ thống có thể chạy ngay lập tức mà không cần huấn luyện lại, bạn cần tải bộ Artifacts (Model & Cache) đã được đóng gói sẵn:
* **Tải xuống tại:** [Kaggle LightGCN Model Artifacts](https://www.kaggle.com/datasets/b22dckh068donngkhoa/lightgcn-model)
* **Danh sách file bắt buộc:**
    * `user_vectors.npy`, `item_vectors.npy`: Embeddings từ mô hình LightGCN.
    * `mappings.db`: Cơ sở dữ liệu SQLite chứa thông tin bài hát (thay thế cho Dictionary khổng lồ để tiết kiệm RAM).
    * `mappings_small.pkl`: Bản đồ ID người dùng và bài hát rút gọn.
    * `master_indexes_cache.pkl`: Các vector nội dung đã tính toán sẵn.
    * `tfidf_model.pkl`, `svd_64_model.pkl`: Mô hình xử lý văn bản.

## 2. Cài đặt Môi trường
Hệ thống yêu cầu Python 3.8+ và các thư viện xử lý ma trận/vector sau:
```bash
pip install scikit-learn scipy pandas numpy tqdm joblib faiss-cpu
```

## 3. Cấu hình Đường dẫn (Quan trọng)
Trong file `main.py` (hoặc script thực thi), bạn cần thay đổi các hằng số đường dẫn trỏ đúng vào thư mục chứa dữ liệu đã tải về. 

**Lưu ý:** Sử dụng dấu gạch chéo xuôi `/` ngay cả trên Windows để tránh lỗi ký tự đặc biệt như `\t` (tab) hay `\n` (newline).

```python
# Đường dẫn thư mục chứa model
LOCAL_MODEL_DIR = 'new_model/tfidf/new'               

# Đường dẫn file SQLite (Tra cứu không tốn RAM)
LOCAL_DB        = 'new_model/tfidf/new/mappings.db'  

# Đường dẫn file pkl rút gọn
LOCAL_SMALL_PKL = 'new_model/tfidf/new/mappings_small.pkl'  

# Thư mục lưu trữ cache tính toán
LOCAL_CACHE_DIR = 'new_model/tfidf/new'
```

## 4. Cách Chạy Demo
Sau khi đã cấu hình xong đường dẫn, bạn chỉ cần thực thi file chính để xem kết quả gợi ý:
```bash
python main.py
```
## 5. Các Tính năng Khuyến nghị Hệ sinh thái

Hệ thống cung cấp một bộ API gợi ý cực kỳ phong phú, đáp ứng mọi kịch bản trải nghiệm nghe nhạc của người dùng thực tế:

* **Hybrid Recommendation (`recommend_hybrid`):** Kết hợp chặt chẽ giữa hành vi người dùng (LightGCN) và nội dung bài hát (TF-IDF) thông qua trọng số $\alpha$ tùy chỉnh, giúp khắc phục triệt để điểm yếu của các mô hình truyền thống.
* **Inclusive Recommendation (`recommend_inclusive`):** Đảm bảo sự công bằng cho các bài hát ít người biết. Tính năng này pha trộn giữa các bài hát thịnh hành (Top Hits) và các bài hát ẩn giấu, chưa từng có lượt nghe (Hidden Gems).
* **Cold Content (`recommend_cold_content`):** Gợi ý thuần túy dựa trên nội dung văn bản (TF-IDF), là "cứu tinh" cho những người dùng mới (Cold-start users) chưa có lịch sử nghe nhạc.
* **Similar to New Item (`recommend_similar_to_new_item`):** Gợi ý ngay lập tức các bài hát tương đồng khi một bài hát mới toanh vừa được thêm vào hệ thống mà **không cần phải huấn luyện lại mô hình**.
* **Playlist Generator (`generate_playlist`):** Tự động sinh danh sách phát bằng cách pha trộn tinh tế giữa "Gu" của người dùng và một (hoặc nhiều) bài hát mồi (Seed tracks) làm điểm tựa.
* **Real-time Session (`recommend_realtime`):** Phản ứng tức thì với ngữ cảnh. Hệ thống sẽ bẻ lái gợi ý dựa trên danh sách các bài hát người dùng vừa nghe liên tiếp trong vài phút qua.
* **Trending (`recommend_trending`):** Thuật toán tính điểm xu hướng theo thời gian bán rã (Half-life), giúp đề xuất các bài hát đang hot nhất toàn hệ thống nhưng vẫn được cá nhân hóa theo từng người dùng.
* **Discovery (`recommend_discovery`):** Gợi ý mang tính đột phá (Serendipity), chủ động kéo người dùng thoát khỏi "vùng an toàn" (Filter Bubble) để khám phá những nghệ sĩ/thể loại mới lạ nhưng vẫn hợp tai.
* **Similar Users (`recommend_similar_users`):** Khai thác sức mạnh cộng đồng bằng cách quét qua không gian vector để tìm những người dùng "tâm giao" có cùng sở thích, từ đó đưa ra gợi ý chéo.
* **Next in Session (`recommend_next_in_session`):** Hoạt động như tính năng "Autoplay", dự đoán bài hát tiếp theo hoàn hảo nhất để phát dựa trên chuỗi bài hát của phiên nghe hiện tại.
* **Timeframe Filtering (`recommend_by_timeframe`):** Phân tích sự thay đổi gu âm nhạc theo thời gian, cho phép lọc và cá nhân hóa gợi ý chỉ dựa trên lịch sử nghe trong một khoảng thời gian cụ thể (ví dụ: 6 tháng qua).

## 6. Đánh giá Hiệu suất (Evaluation Metrics)

Hệ thống đã được đánh giá thực tế trên tập dữ liệu kiểm thử (Test set) với thang đo **K = 20**. Kết quả cho thấy việc kết hợp nội dung văn bản (TF-IDF) vào mô hình học sâu đồ thị (LightGCN) mang lại sự cải thiện rõ rệt về mọi mặt.

### 6.1. Bảng so sánh độ đo định lượng

| Metric (K=20) | Base LightGCN ($\alpha=0.0$) | Hybrid TF-IDF ($\alpha=0.25$) | Mức cải thiện |
| :--- | :---: | :---: | :---: |
| **Recall@20** | 0.0405 | **0.0499** | **+ 23.2%** |
| **Precision@20** | 0.0301 | **0.0357** | **+ 18.6%** |
| **NDCG@20** | 0.0483 | **0.0597** | **+ 23.6%** |

* **Nhận xét:** Việc áp dụng trọng số Hybrid $\alpha=0.25$ (75% hành vi Collaborative + 25% độ tương đồng Content) giúp mô hình gợi ý chính xác hơn đáng kể. Chỉ số **NDCG tăng 23.6%** cho thấy các bài hát mà người dùng thực sự muốn nghe đã được đẩy lên các thứ hạng cao hơn trong Top 20. 
* **Đánh đổi (Trade-off):** Tốc độ suy luận của mô hình Hybrid chậm hơn một chút so với Base LightGCN thuần (1.60s/batch so với ~0.4s/batch) do phải tính toán thêm ma trận nhúng văn bản. Tuy nhiên, mức thời gian này vẫn hoàn toàn đáp ứng tốt cho hệ thống Local.

### 6.2. Đánh giá định tính qua các kịch bản (Qualitative Results)
Dựa trên log chạy thực tế với một người dùng ngẫu nhiên (`User=13`, Tier: Warm), hệ thống chứng minh khả năng linh hoạt cao:
* **Giải quyết Cold-Start & Hidden Gems:** Thuật toán `recommend_inclusive` xuất sắc trong việc pha trộn các bài hát thịnh hành (Top Hits từ mô hình đồ thị) và các bài hát chưa từng xuất hiện trong mô hình nhưng có metadata tương đồng (Hidden Gems khai thác bởi TF-IDF).
* **Bắt kịp ngữ cảnh (Session-based):** Mô hình Real-time và Next-in-Session phản ứng cực nhạy. Khi user đưa vào 3 bài mồi (Seed tracks), hệ thống lập tức hướng các kết quả tiếp theo sang các nghệ sĩ tương tự (như *The Chemical Brothers*, *The Crystal Method*, *Metallica*).
* **Khám phá (Serendipity):** Chế độ `recommend_discovery` tự động đẩy người dùng ra khỏi "vùng an toàn" để tìm kiếm các bài hát/nghệ sĩ mới nhưng vẫn giữ được sợi dây liên kết vô hình với sở thích gốc.

Music User Recommendation & Analytics

Dự án này tập trung vào việc khai phá dữ liệu nghe nhạc(từ ListenBrain) để thấu hiểu hành vi người dùng và cung cấp các gợi ý cá nhân hóa chính xác nhất. Hệ thống được chia thành 4 nhánh (branch) tương ứng với 4 giai đoạn phân tích quan trọng.

Cấu trúc các Branch

Dưới đây là mô tả chi tiết nhiệm vụ của từng nhánh phát triển dựa trên tiến độ commit:

Module 1: Phân cụm (Clustering)	Phân loại User dựa trên hành vi nghe nhạc. Xác định các chân dung khách hàng như: "Cú đêm", "Fan cứng", hay "Người mới".

Module 2: Gu đại chúng (Niche Mining)	Phân tích sâu về khẩu vị âm nhạc. Trả lời câu hỏi: User này là người nghe nhạc dẫn đầu xu hướng (Mainstream) hay là người thích nghe nhạc ngách (Niche)?

Module 3: Hệ khuyến nghị (RecSys)	Triển khai thuật toán ALS (Alternating Least Squares) cho Implicit Collaborative Filtering. Gợi ý bài hát dựa trên toàn bộ lịch sử nghe nhạc với cơ chế lọc tin cậy.

Module 4: Churn & Dashboard	Dự báo tỷ lệ người dùng rời bỏ ứng dụng (Churn Prediction). Tích hợp toàn bộ kết quả từ các module trước lên Web Dashboard trực quan.

Link phân công công việc : https://docs.google.com/spreadsheets/d/11epef7jHn0fEMzOTHz0TlGQOkU2Z5Lff1clV7m3yo8g/edit?gid=0#gid=0

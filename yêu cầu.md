2. Luồng xử lý AI (Backend)
Đây là flow xử lý chính khi bệnh nhân upload ảnh:
Input: Hình ảnh vết loét từ bệnh nhân.
Step 1: Segmentation (Phân đoạn)
Sử dụng model SegformerB3 (từ file .pth) để tạo segmentation mask (xác định vùng vết loét).
Step 2: Size Analysis (Phân tích Kích thước)
Dựa vào mask trong tap train/valid, tính toán kích thước/diện tích của vết loét
Step 3: Color Analysis (Phân tích Màu sắc)
Sử dụng K-means (lần 1) để tìm ra số K tối ưu (số cụm màu đặc trưng) trong vùng vết loét.
Sử dụng K-means (lần 2) với số K đã tìm được để phân cụm màu sắc (ví dụ: mô hoại tử, mô hạt, v.v.).
Step 4: Roughness Analysis (Phân tích Bề mặt)
Áp dụng "Roughness Qualification" (phân tích độ gồ ghề/kết cấu) trên vùng mask.
Step 5: Risk Level (Đánh giá Rủi ro)
Dùng K-means (hoặc 1 mô hình khác) để tổng hợp các yếu tố (kích thước, màu, độ nhám) và đưa ra mức độ/level rủi ro của vết loét.
Output (Backend):
Dữ liệu phân tích (JSON, text).
Hình ảnh visualize (plot) đã được xử lý.

3. Chức năng Web App (Frontend & Roles)
A. Bệnh nhân (Patient)
Authentication: Đăng ký / Đăng nhập.
Upload: Chụp hoặc tải ảnh vết loét lên hệ thống.
Trigger Analysis: Nút "Phân tích" để gửi ảnh qua pipeline AI (Mục 2).
View Results: Xem kết quả phân tích trực quan (visualize plot) ngay trên web.
Export PDF:
Tự động tạo file PDF báo cáo kết quả.
PDF phải bao gồm: ID Bệnh nhân, hình ảnh, các chỉ số phân tích (kích thước, % màu sắc, mức độ rủi ro).
Cho phép bệnh nhân tải về (để gửi cho bác sĩ).
AI Chatbot: 
Một cửa sổ chat (giống Tawk.to hoặc Messenger) để bệnh nhân gõ câu hỏi.
Chatbot trả lời các câu hỏi chung (ví dụ: "Bệnh tiểu đường là gì?", "Cách chăm sóc vết loét cơ bản?", "Kết quả này nghĩa là gì?").
Chat với Bác sĩ (1-1): 
Kênh chat riêng (có thể trong trang quản lý của Bác sĩ) để nhắn tin, gửi thêm hình ảnh, và nhận tư vấn trực tiếp từ Bác sĩ của mình.

B. Bác sĩ (Doctor)
Authentication: Đăng nhập (tài khoản do Admin cấp).
Patient Search: Trang "Tìm kiếm Bệnh nhân" (dựa trên ID Bệnh nhân mà họ cung cấp).
View Analysis: Xem lại toàn bộ lịch sử phân tích (hình ảnh, plots, PDF) của bệnh nhân đó.
Add Recommendation: Thêm "Khuyến nghị" hoặc "Chỉ định" cho bệnh nhân dựa trên kết quả phân tích.
Chat với Bệnh nhân (1-1): 
Dashboard quản lý các cuộc hội thoại với bệnh nhân.
Nhận và trả lời tin nhắn (text, hình ảnh) từ bệnh nhân.
C. Admin
Authentication: Đăng nhập Admin.
User Management: Quản lý tài khoản (tạo, khóa, reset password) cho Bác sĩ và Bệnh nhân.
Appointment Management: Quản lý lịch khám (tạo, xem, hủy lịch).
(Optional): Xem thống kê tổng quan (số lượt dùng, số bệnh nhân).
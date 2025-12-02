# 🗄️ Render Database Setup Guide

## Vấn đề
Backend đã deploy lên Render nhưng database chưa có tables và users, nên không thể đăng nhập/đăng ký.

## Giải pháp

### Bước 1: Đợi Backend Deploy Xong
1. Vào https://dashboard.render.com
2. Chọn service **wound-care-backend**
3. Đợi deploy hoàn tất (status: **Live**)

### Bước 2: Khởi tạo Database
1. Trong dashboard của **wound-care-backend**, click tab **Shell**
2. Chạy lệnh sau:
```bash
python init_db.py
```

3. Đợi script chạy xong, bạn sẽ thấy:
```
✅ Database initialization completed successfully!

📝 Test Accounts:
   Admin:   admin@woundcare.ai / admin123
   Doctor:  doctor@woundcare.ai / doctor123
   Patient: patient@woundcare.ai / patient123
```

### Bước 3: Test Login
1. Vào frontend: https://wound-care-ai.vercel.app
2. Đăng nhập với một trong các tài khoản test:
   - **Admin**: admin@woundcare.ai / admin123
   - **Doctor**: doctor@woundcare.ai / doctor123
   - **Patient**: patient@woundcare.ai / patient123

## Google OAuth Setup

### Bước 1: Cấu hình Google OAuth
1. Vào https://console.cloud.google.com
2. Chọn project của bạn
3. Vào **APIs & Services** > **Credentials**
4. Chọn OAuth 2.0 Client ID
5. Thêm **Authorized redirect URIs**:
   - `https://wound-care-ai.vercel.app/auth/callback`
   - `https://wound-care-ai.vercel.app`

### Bước 2: Cập nhật Environment Variables trên Render
1. Vào https://dashboard.render.com
2. Chọn service **wound-care-backend**
3. Click tab **Environment**
4. Thêm/cập nhật:
   - `GOOGLE_CLIENT_ID`: [Your Google Client ID]
   - `GOOGLE_CLIENT_SECRET`: [Your Google Client Secret]
5. Click **Save Changes** (service sẽ tự động restart)

### Bước 3: Cập nhật Environment Variables trên Vercel
1. Vào https://vercel.com/dashboard
2. Chọn project **wound-care-ai**
3. Vào **Settings** > **Environment Variables**
4. Thêm/cập nhật:
   - `REACT_APP_GOOGLE_CLIENT_ID`: [Your Google Client ID]
   - `REACT_APP_API_URL`: https://wound-care-backend.onrender.com
5. Redeploy frontend

## Troubleshooting

### Lỗi: "Cannot import 'setuptools.build_meta'"
- **Nguyên nhân**: Python version không đúng
- **Giải pháp**: Đã fix bằng file `.python-version` và cập nhật `render.yaml`

### Lỗi: "Database connection failed"
- **Nguyên nhân**: Database chưa được tạo
- **Giải pháp**: Render tự động tạo database từ `render.yaml`, chỉ cần chạy `init_db.py`

### Lỗi: "Table doesn't exist"
- **Nguyên nhân**: Chưa chạy init_db.py
- **Giải pháp**: Chạy `python init_db.py` trong Render Shell

### Google OAuth không hoạt động
- **Nguyên nhân**: Redirect URI chưa được cấu hình
- **Giải pháp**: Thêm redirect URIs trong Google Console như hướng dẫn trên

## Kiểm tra Database

Để kiểm tra database đã có data chưa, chạy trong Render Shell:

```bash
python -c "
from database import get_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM users')
count = cursor.fetchone()[0]
print(f'Total users: {count}')
cursor.close()
conn.close()
"
```

## Tạo User Mới Thủ Công

Nếu cần tạo user mới, chạy trong Render Shell:

```bash
python -c "
from database import get_db_connection
from werkzeug.security import generate_password_hash

conn = get_db_connection()
cursor = conn.cursor()

email = 'newuser@example.com'
password = generate_password_hash('password123')
full_name = 'New User'
role = 'patient'  # hoặc 'doctor', 'admin'

cursor.execute('''
    INSERT INTO users (email, password_hash, full_name, role, is_active)
    VALUES (%s, %s, %s, %s, %s)
''', (email, password, full_name, role, True))

conn.commit()
print(f'Created user: {email}')

cursor.close()
conn.close()
"
```

## Xóa và Tạo Lại Database

Nếu cần reset database hoàn toàn:

```bash
python init_db.py
```

Script này sẽ tự động:
1. Drop tất cả tables cũ
2. Tạo lại tables mới
3. Insert test users

## Lưu ý

- Database trên Render (Free tier) có giới hạn 1GB storage
- Database sẽ bị xóa sau 90 ngày không hoạt động
- Nên backup database định kỳ nếu có data quan trọng

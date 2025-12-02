# 🚀 Quick Deploy Guide

## Chuẩn Bị

Đảm bảo bạn đã:
- ✅ Cài Render CLI: `npm install -g @render-com/cli`
- ✅ Cài Vercel CLI: `npm install -g vercel`
- ✅ Đăng nhập Render: `render login`
- ✅ Đăng nhập Vercel: `vercel login`
- ✅ Push code lên GitHub

---

## Cách 1: Deploy Tự Động (Khuyến nghị)

```bash
./deploy-all.sh
```

Script sẽ hướng dẫn bạn từng bước!

---

## Cách 2: Deploy Thủ Công

### A. Deploy Backend lên Render

1. **Tạo Web Service trên Render**
   - Vào https://dashboard.render.com
   - Click **New +** → **Web Service**
   - Connect GitHub repo

2. **Cấu hình Service**
   ```
   Name: wound-care-backend
   Root Directory: wound-care-ai/backend
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app
   ```

3. **Thêm Environment Variables**
   ```
   SECRET_KEY=<generate-random-32-chars>
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   MODEL_PATH=model_files/segformer_wound.pth
   DATASET_PATH=../../Model/wound_features_with_risk.csv
   COLOR_DATASET_PATH=../../Model/color_features_ulcer_red_yellow_dark.csv
   GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
   GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET
   FE_URL=https://wound-care-ai.vercel.app
   ```

4. **Thêm Database**
   - Click **New +** → **PostgreSQL**
   - Hoặc dùng MySQL external
   - Copy DATABASE_URL vào environment variables

5. **Deploy!**
   - Click **Create Web Service**
   - Đợi 5-10 phút
   - Copy URL backend (vd: `https://wound-care-backend.onrender.com`)

### B. Deploy Frontend lên Vercel

1. **Tạo file .env.production**
   ```bash
   cd wound-care-ai/frontend
   echo "REACT_APP_API_URL=https://wound-care-backend.onrender.com/api" > .env.production
   ```

2. **Deploy**
   ```bash
   vercel --prod
   ```

3. **Hoặc dùng Vercel Dashboard**
   - Vào https://vercel.com/new
   - Import GitHub repo
   - Root Directory: `wound-care-ai/frontend`
   - Framework: Create React App
   - Environment Variables:
     ```
     REACT_APP_API_URL=https://wound-care-backend.onrender.com/api
     ```
   - Deploy!

---

## Sau Khi Deploy

### 1. Update Google OAuth

Vào https://console.cloud.google.com/apis/credentials

**Authorized JavaScript origins:**
```
https://wound-care-ai.vercel.app
https://wound-care-backend.onrender.com
```

**Authorized redirect URIs:**
```
https://wound-care-ai.vercel.app/auth/callback
https://wound-care-backend.onrender.com/api/auth/callback
```

### 2. Update Backend Environment

Vào Render Dashboard → Environment:
```
FE_URL=https://wound-care-ai.vercel.app
BE_URL=https://wound-care-backend.onrender.com
```

### 3. Setup Database

**Option A: PostgreSQL trên Render (Khuyến nghị)**
```bash
# Render tự động tạo DATABASE_URL
# Chỉ cần import schema
```

**Option B: MySQL External**
```bash
# Update DATABASE_URL trong Render:
DATABASE_URL=mysql+mysqlconnector://user:pass@host:3306/dbname
```

### 4. Import Database Schema

```bash
# Nếu dùng PostgreSQL, convert schema từ MySQL sang PostgreSQL
# Hoặc dùng MySQL external và import trực tiếp
```

---

## Test Deployment

1. **Test Frontend**: https://wound-care-ai.vercel.app
2. **Test Backend**: https://wound-care-backend.onrender.com/api/health
3. **Test Login**: Thử đăng nhập bằng Google

---

## Troubleshooting

### Backend không chạy
- Check logs: Render Dashboard → Logs
- Verify environment variables
- Check DATABASE_URL format

### Frontend không connect được backend
- Verify REACT_APP_API_URL
- Check CORS settings
- Verify backend URL

### Google OAuth lỗi
- Verify redirect URIs
- Check client ID/secret
- Clear browser cache

---

## Update Code

### Backend
```bash
git push origin main
# Render tự động deploy
```

### Frontend
```bash
git push origin main
# Vercel tự động deploy
```

Hoặc deploy thủ công:
```bash
cd wound-care-ai/frontend
vercel --prod
```

---

## Chi Phí

- **Render**: Free tier (đủ dùng) hoặc $7/tháng (Starter)
- **Vercel**: Free (đủ dùng)
- **Database**: 
  - PostgreSQL trên Render: Free 90 days, sau đó $7/tháng
  - MySQL external: $5-10/tháng

**Tổng**: $0-17/tháng

---

## Support

Nếu gặp vấn đề:
1. Check logs trên Render/Vercel dashboard
2. Verify environment variables
3. Test API endpoints
4. Check Google OAuth settings

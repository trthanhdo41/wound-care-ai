#!/bin/bash

echo "🚀 Wound Care AI - Step by Step Deployment"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}STEP 1: Deploy Backend to Render${NC}"
echo "-----------------------------------"
echo ""
echo "1. Go to: https://dashboard.render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repo"
echo "4. Settings:"
echo "   Name: wound-care-backend"
echo "   Root Directory: wound-care-ai/backend"
echo "   Environment: Python 3"
echo "   Build Command: pip install -r requirements.txt"
echo "   Start Command: gunicorn --bind 0.0.0.0:\$PORT --workers 2 --timeout 120 app:app"
echo ""
echo "5. Environment Variables (copy paste này):"
echo ""
cat << 'EOF'
SECRET_KEY=eb65c56bbb0c2b9761073f241ada7555037aaa9cc5962d1bb062ca9d027ce9a3
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
MODEL_PATH=model_files/segformer_wound.pth
DATASET_PATH=../../Model/wound_features_with_risk.csv
COLOR_DATASET_PATH=../../Model/color_features_ulcer_red_yellow_dark.csv
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET
FE_URL=https://wound-care-ai.vercel.app
EOF
echo ""
echo "6. Add PostgreSQL Database:"
echo "   - Click 'New +' → 'PostgreSQL'"
echo "   - Name: wound-care-db"
echo "   - Link to your web service"
echo "   - DATABASE_URL will be auto-added"
echo ""
echo -e "${YELLOW}⏳ Wait for backend to deploy (5-10 minutes)...${NC}"
echo ""
read -p "Enter your backend URL when ready (e.g., https://wound-care-backend.onrender.com): " BACKEND_URL

if [ -z "$BACKEND_URL" ]; then
    echo -e "${RED}❌ Backend URL is required!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Backend URL saved: $BACKEND_URL${NC}"
echo ""

# STEP 2: Deploy Frontend
echo ""
echo -e "${YELLOW}STEP 2: Deploy Frontend to Vercel${NC}"
echo "-----------------------------------"
echo ""

cd wound-care-ai/frontend

# Create .env.production
echo "Creating .env.production..."
cat > .env.production << EOF
REACT_APP_API_URL=${BACKEND_URL}/api
EOF

echo -e "${GREEN}✅ Created .env.production with API URL: ${BACKEND_URL}/api${NC}"
echo ""

# Check if .vercel exists (already linked)
if [ -d ".vercel" ]; then
    echo "Project already linked to Vercel. Deploying..."
    vercel --prod
else
    echo "First time deployment. Follow Vercel prompts..."
    echo ""
    echo "When asked:"
    echo "  - Set up and deploy? → YES"
    echo "  - Link to existing project? → NO"
    echo "  - Project name? → wound-care-ai"
    echo "  - Directory? → ./"
    echo ""
    vercel --prod
fi

cd ../..

echo ""
echo -e "${YELLOW}STEP 3: Get Frontend URL${NC}"
echo "-----------------------------------"
echo ""
read -p "Enter your Vercel URL (from output above, e.g., https://wound-care-ai.vercel.app): " FRONTEND_URL

if [ -z "$FRONTEND_URL" ]; then
    FRONTEND_URL="https://wound-care-ai.vercel.app"
    echo "Using default: $FRONTEND_URL"
fi

echo ""
echo -e "${GREEN}✅ Frontend URL saved: $FRONTEND_URL${NC}"
echo ""

# STEP 3: Post-deployment
echo ""
echo -e "${YELLOW}STEP 4: Update Configurations${NC}"
echo "-----------------------------------"
echo ""
echo "📋 TODO List:"
echo ""
echo "1️⃣  Update Google OAuth (https://console.cloud.google.com):"
echo "   Authorized JavaScript origins:"
echo "   - ${FRONTEND_URL}"
echo "   - ${BACKEND_URL}"
echo ""
echo "   Authorized redirect URIs:"
echo "   - ${FRONTEND_URL}/auth/callback"
echo "   - ${BACKEND_URL}/api/auth/callback"
echo ""
echo "2️⃣  Update Render Environment Variables:"
echo "   Go to: ${BACKEND_URL} → Environment"
echo "   Update:"
echo "   - FE_URL=${FRONTEND_URL}"
echo "   - BE_URL=${BACKEND_URL}"
echo ""
echo "3️⃣  Test your app:"
echo "   - Frontend: ${FRONTEND_URL}"
echo "   - Backend Health: ${BACKEND_URL}/api/health"
echo ""

# Summary
echo ""
echo -e "${GREEN}=========================================="
echo "✅ Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "🌐 Frontend: ${FRONTEND_URL}"
echo "🔧 Backend: ${BACKEND_URL}"
echo ""
echo "📝 Save these URLs for future reference!"
echo ""

#!/bin/bash

echo "🚀 Deploying Wound Care AI to Production"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Deploy Backend to Render
echo -e "${BLUE}Step 1: Deploying Backend to Render...${NC}"
echo ""
echo "Please follow these steps:"
echo "1. Go to https://dashboard.render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Configure:"
echo "   - Name: wound-care-backend"
echo "   - Root Directory: wound-care-ai/backend"
echo "   - Environment: Python 3"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: gunicorn --bind 0.0.0.0:\$PORT --workers 2 --timeout 120 app:app"
echo ""
echo "5. Add Environment Variables:"
echo "   DATABASE_URL: (will be auto-generated when you add PostgreSQL)"
echo "   SECRET_KEY: $(openssl rand -hex 32)"
echo "   ALGORITHM: HS256"
echo "   ACCESS_TOKEN_EXPIRE_MINUTES: 30"
echo "   MODEL_PATH: model_files/segformer_wound.pth"
echo "   DATASET_PATH: ../../Model/wound_features_with_risk.csv"
echo "   COLOR_DATASET_PATH: ../../Model/color_features_ulcer_red_yellow_dark.csv"
echo "   GOOGLE_CLIENT_ID: YOUR_GOOGLE_CLIENT_ID"
echo "   GOOGLE_CLIENT_SECRET: YOUR_GOOGLE_CLIENT_SECRET"
echo "   FE_URL: https://wound-care-ai.vercel.app"
echo ""
echo ""
read -p "Press Enter when backend is deployed and you have the URL..."
echo ""
read -p "Enter your Render backend URL (e.g., https://wound-care-backend.onrender.com): " BACKEND_URL

# Validate URL
if [ -z "$BACKEND_URL" ]; then
    echo -e "${RED}❌ Backend URL is required!${NC}"
    exit 1
fi

# Step 2: Deploy Frontend to Vercel
echo ""
echo -e "${BLUE}Step 2: Deploying Frontend to Vercel...${NC}"
cd wound-care-ai/frontend

# Create production env
cat > .env.production << EOF
REACT_APP_API_URL=${BACKEND_URL}/api
EOF

echo "Environment file created with API URL: ${BACKEND_URL}/api"
echo ""
echo "Deploying to Vercel..."

# Deploy with environment variable
vercel --prod -e REACT_APP_API_URL="${BACKEND_URL}/api"

# Get frontend URL
echo ""
read -p "Enter your Vercel frontend URL (from output above): " FRONTEND_URL

if [ -z "$FRONTEND_URL" ]; then
    FRONTEND_URL="https://wound-care-ai.vercel.app"
    echo "Using default: $FRONTEND_URL"
fi

cd ../..

# Step 3: Update configurations
echo ""
echo -e "${BLUE}Step 3: Post-deployment configuration${NC}"
echo ""
echo "⚠️  IMPORTANT: Update these settings:"
echo ""
echo "1. Google OAuth Console (https://console.cloud.google.com):"
echo "   - Add to Authorized JavaScript origins:"
echo "     ${FRONTEND_URL}"
echo "     ${BACKEND_URL}"
echo "   - Add to Authorized redirect URIs:"
echo "     ${FRONTEND_URL}/auth/callback"
echo "     ${BACKEND_URL}/api/auth/callback"
echo ""
echo "2. Render Dashboard:"
echo "   - Update FE_URL environment variable to: ${FRONTEND_URL}"
echo "   - Update BE_URL environment variable to: ${BACKEND_URL}"
echo ""
echo "3. Database Setup:"
echo "   - In Render dashboard, add PostgreSQL database"
echo "   - Or use external MySQL (update DATABASE_URL)"
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
echo "📝 Next steps:"
echo "1. Test the application"
echo "2. Upload model files to Render (if needed)"
echo "3. Import database schema"
echo "4. Create test users"
echo ""
echo "🆘 If you encounter issues:"
echo "- Check Render logs: https://dashboard.render.com"
echo "- Check Vercel logs: https://vercel.com/dashboard"
echo ""

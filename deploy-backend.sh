#!/bin/bash

echo "🚀 Deploying Backend to Render..."

# Check if render CLI is installed
if ! command -v render &> /dev/null; then
    echo "❌ Render CLI not found. Installing..."
    npm install -g @render-com/cli
fi

# Login to Render (if not already logged in)
echo "📝 Checking Render authentication..."
render whoami || render login

# Deploy backend
echo "🔨 Deploying backend service..."
cd wound-care-ai/backend

# Create .env for Render (will be set via dashboard)
cat > .env.render << EOF
# These will be set in Render dashboard
DATABASE_URL=\${DATABASE_URL}
SECRET_KEY=\${SECRET_KEY}
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
MODEL_PATH=model_files/segformer_wound.pth
DATASET_PATH=../../Model/wound_features_with_risk.csv
COLOR_DATASET_PATH=../../Model/color_features_ulcer_red_yellow_dark.csv
GOOGLE_CLIENT_ID=YOUR_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=YOUR_GOOGLE_CLIENT_SECRET
FE_URL=https://wound-care-ai.vercel.app
BE_URL=\${RENDER_EXTERNAL_URL}
EOF

echo "✅ Backend deployment initiated!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://dashboard.render.com"
echo "2. Create new Web Service"
echo "3. Connect your GitHub repo"
echo "4. Select 'wound-care-ai/backend' as root directory"
echo "5. Set environment variables from .env.render"
echo "6. Click 'Create Web Service'"
echo ""
echo "🔗 After deploy, your backend will be at: https://wound-care-backend.onrender.com"

cd ../..

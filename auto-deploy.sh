#!/bin/bash

echo "🚀 Auto Deploy to Render + Vercel"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Render CLI
if ! command -v render &> /dev/null; then
    echo -e "${RED}❌ Render CLI not found${NC}"
    echo "Install: npm install -g @render-com/cli"
    exit 1
fi

# Check Vercel CLI
if ! command -v vercel &> /dev/null; then
    echo -e "${RED}❌ Vercel CLI not found${NC}"
    echo "Install: npm install -g vercel"
    exit 1
fi

echo -e "${BLUE}Step 1: Deploy Backend to Render${NC}"
echo "-----------------------------------"
echo ""

# Check if logged in
render whoami > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Please login to Render..."
    render login
fi

echo "Creating Render service from render.yaml..."
echo ""
echo "⚠️  You need to set these secrets in Render Dashboard:"
echo "   GOOGLE_CLIENT_ID=<your-google-client-id>"
echo "   GOOGLE_CLIENT_SECRET=<your-google-client-secret>"
echo ""

# Deploy using render.yaml
cd wound-care-ai
render blueprint launch

echo ""
echo -e "${YELLOW}⏳ Waiting for backend to deploy...${NC}"
echo "Check status at: https://dashboard.render.com"
echo ""
read -p "Enter your Render backend URL when ready: " BACKEND_URL

if [ -z "$BACKEND_URL" ]; then
    echo -e "${RED}❌ Backend URL required!${NC}"
    exit 1
fi

cd ..

echo ""
echo -e "${BLUE}Step 2: Deploy Frontend to Vercel${NC}"
echo "-----------------------------------"
echo ""

cd wound-care-ai/frontend

# Create .env.production
cat > .env.production << EOF
REACT_APP_API_URL=${BACKEND_URL}/api
EOF

echo "✅ Created .env.production"
echo ""

# Check if logged in
vercel whoami > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Please login to Vercel..."
    vercel login
fi

echo "Deploying to Vercel..."
vercel --prod

cd ../..

echo ""
echo -e "${GREEN}=========================================="
echo "✅ Deployment Complete!"
echo "==========================================${NC}"
echo ""
echo "🔗 Backend: ${BACKEND_URL}"
echo "🔗 Frontend: Check Vercel output above"
echo ""
echo "📝 Next steps:"
echo "1. Update Google OAuth redirect URIs"
echo "2. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in Render"
echo "3. Test your application"
echo ""

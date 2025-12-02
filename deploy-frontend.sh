#!/bin/bash

echo "🚀 Deploying Frontend to Vercel..."

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Login to Vercel (if not already logged in)
echo "📝 Checking Vercel authentication..."
vercel whoami || vercel login

# Deploy frontend
echo "🔨 Deploying frontend..."
cd wound-care-ai/frontend

# Create .env.production
echo "Creating production environment variables..."
cat > .env.production << EOF
REACT_APP_API_URL=https://wound-care-backend.onrender.com/api
EOF

# Deploy to Vercel
echo "📤 Deploying to Vercel..."
vercel --prod

echo ""
echo "✅ Frontend deployed successfully!"
echo "🔗 Your app is live at: https://wound-care-ai.vercel.app"
echo ""
echo "⚠️  Don't forget to:"
echo "1. Update Google OAuth redirect URIs with your Vercel domain"
echo "2. Update FE_URL in Render backend environment variables"

cd ../..

#!/bin/bash

echo "🚀 Deploying Backend to Render..."

# Add and commit changes
echo "📝 Committing latest changes..."
git add wound-care-ai/backend/requirements.txt
git add wound-care-ai/render.yaml
git commit -m "Fix: Add setuptools and wheel to requirements for Python 3.11 compatibility"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main

echo "✅ Code pushed to GitHub!"
echo ""
echo "🔄 Render will automatically detect the changes and redeploy."
echo "📊 Monitor deployment at: https://dashboard.render.com"
echo ""
echo "⏱️  Deployment usually takes 5-10 minutes."
echo "🔗 Your backend will be at: https://wound-care-backend.onrender.com"

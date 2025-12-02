#!/bin/bash

echo "📦 Setting up Git repository..."
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    echo -e "${GREEN}✅ Git initialized${NC}"
else
    echo -e "${YELLOW}⚠️  Git already initialized${NC}"
fi

# Create .gitignore if not exists
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
.pytest_cache/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*

# React
/build
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment
.env
*.env

# Database
*.db
*.sqlite
*.sqlite3

# Uploads
uploads/
test_output/

# Models (too large for git)
*.pth
*.pkl
*.h5
*.onnx

# Logs
*.log
logs/

# Deployment
.vercel
.render

# OS
Thumbs.db
EOF
    echo -e "${GREEN}✅ Created .gitignore${NC}"
else
    echo -e "${YELLOW}⚠️  .gitignore already exists${NC}"
fi

echo ""
echo -e "${BLUE}Current status:${NC}"
git status

echo ""
echo -e "${YELLOW}Do you want to commit all files? (y/n)${NC}"
read -p "> " COMMIT_CHOICE

if [ "$COMMIT_CHOICE" = "y" ] || [ "$COMMIT_CHOICE" = "Y" ]; then
    echo ""
    echo "Adding files to git..."
    git add .
    
    echo ""
    echo "Committing..."
    git commit -m "Initial commit: Wound Care AI application"
    
    echo -e "${GREEN}✅ Files committed${NC}"
else
    echo "Skipping commit..."
fi

echo ""
echo -e "${YELLOW}Do you have a GitHub repository URL? (y/n)${NC}"
read -p "> " HAS_REPO

if [ "$HAS_REPO" = "y" ] || [ "$HAS_REPO" = "Y" ]; then
    echo ""
    read -p "Enter your GitHub repository URL (e.g., https://github.com/username/wound-care-ai.git): " REPO_URL
    
    if [ ! -z "$REPO_URL" ]; then
        echo ""
        echo "Adding remote origin..."
        git remote add origin "$REPO_URL" 2>/dev/null || git remote set-url origin "$REPO_URL"
        
        echo ""
        echo "Pushing to GitHub..."
        git branch -M main
        git push -u origin main
        
        echo ""
        echo -e "${GREEN}✅ Code pushed to GitHub!${NC}"
        echo ""
        echo "🔗 Repository: $REPO_URL"
    fi
else
    echo ""
    echo -e "${YELLOW}📋 To create a GitHub repository:${NC}"
    echo ""
    echo "1. Go to: https://github.com/new"
    echo "2. Repository name: wound-care-ai"
    echo "3. Description: AI-powered diabetic foot ulcer analysis system"
    echo "4. Visibility: Private (recommended) or Public"
    echo "5. Don't initialize with README, .gitignore, or license"
    echo "6. Click 'Create repository'"
    echo ""
    echo "Then run these commands:"
    echo ""
    echo "  git remote add origin https://github.com/YOUR_USERNAME/wound-care-ai.git"
    echo "  git branch -M main"
    echo "  git push -u origin main"
    echo ""
fi

echo ""
echo -e "${GREEN}=========================================="
echo "✅ Git Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "📝 Next steps:"
echo "1. Make sure code is pushed to GitHub"
echo "2. Run: ./deploy-step-by-step.sh"
echo ""

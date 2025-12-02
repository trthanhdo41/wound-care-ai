#!/bin/bash

echo "🗄️  Setting up Render Database..."
echo ""
echo "This script will initialize your database on Render."
echo ""

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL environment variable not set"
    echo ""
    echo "Please run this command on Render Shell:"
    echo "1. Go to https://dashboard.render.com"
    echo "2. Select your backend service"
    echo "3. Click 'Shell' tab"
    echo "4. Run: python init_db.py"
    exit 1
fi

# Run initialization
python init_db.py

echo ""
echo "✅ Database setup complete!"

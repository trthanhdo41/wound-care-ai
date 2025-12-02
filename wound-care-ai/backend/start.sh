#!/bin/bash
# Render startup script

echo "🚀 Starting Wound Care AI Backend..."

# Run with Gunicorn for production
gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app

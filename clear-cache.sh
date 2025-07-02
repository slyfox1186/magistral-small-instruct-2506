#!/bin/bash

# Clear all caches for the Neural Consciousness Chat System

echo "🧹 Clearing all development caches..."

# Frontend cache clearing
echo "📱 Clearing frontend caches..."
cd client

# Clear Vite cache
echo "  - Clearing Vite cache..."
rm -rf .vite
rm -rf node_modules/.vite

# Clear dist folder
echo "  - Clearing build artifacts..."
rm -rf dist

# Clear browser cache files
echo "  - Clearing local storage cache..."
rm -rf .cache

echo "✅ Frontend caches cleared!"

# Go back to root
cd ..

# Backend cache clearing
echo "🖥️ Clearing backend caches..."

# Clear Python cache
echo "  - Clearing Python __pycache__..."
find backend -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find backend -name "*.pyc" -delete 2>/dev/null || true

# Clear model cache if exists
echo "  - Clearing model cache..."
rm -rf backend/.cache 2>/dev/null || true

echo "✅ Backend caches cleared!"

# Redis cache clearing (optional)
echo "🔴 Clearing Redis cache..."
redis-cli FLUSHALL 2>/dev/null && echo "  - Redis cache cleared!" || echo "  - Redis not available or already clear"

echo ""
echo "🚀 All caches cleared! Ready for fresh development."
echo "   Run: python start.py"
echo "   Or:  cd client && npm run dev:clean"
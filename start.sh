#!/bin/bash
set -euo pipefail

echo "🚀 Starting Sister Products API..."
echo "📋 Database credentials are configured in docker-compose.yml"

# Start the Docker services
docker compose up -d --build

echo "⏳ Waiting for service to be ready..."
sleep 10

# Test health endpoint
echo "🔍 Testing health endpoint..."
if curl -sf http://localhost:8000/health > /dev/null; then
  echo "✅ API is healthy!"
  echo "🌐 API available at: http://localhost:8000"
  echo "📚 API docs at: http://localhost:8000/docs"
else
  echo "❌ Health check failed. Check logs with: docker compose logs api"
fi

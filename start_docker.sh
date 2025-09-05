#!/bin/bash
echo "🚀 Starting Sister Products Mapping System with Docker"
docker-compose up -d --build
echo "✅ Services started!"
echo "🌐 Web interface: http://localhost:8000"
echo "📊 API docs: http://localhost:8000/docs"
echo "💚 Health check: http://localhost:8000/health"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: docker-compose down"

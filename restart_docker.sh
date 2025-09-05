#!/bin/bash
echo "🔄 Restarting Sister Products Mapping System"
docker-compose down
docker-compose up -d --build
echo "✅ Services restarted!"
echo "🌐 Web interface: http://localhost:8000"

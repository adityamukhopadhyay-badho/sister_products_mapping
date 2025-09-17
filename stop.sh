#!/bin/bash
set -euo pipefail

echo "🛑 Stopping Sister Products API..."
docker compose down

echo "✅ Service stopped successfully"

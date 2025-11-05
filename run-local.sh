#!/bin/bash
# Darwin Core 2025 - Local Development Runner

cd "$(dirname "$0")"

echo "🚀 Starting Darwin Core 2025.1.0 (Local Development)"
echo ""

# Export environment variables
export HOST=0.0.0.0
export PORT=8090
export QDRANT_URL=http://localhost:6333
export REDIS_URL=redis://localhost:6379
export OLLAMA_URL=http://localhost:11434
export DARWIN_API_TOKEN=darwin_local_dev_token_2025
export DARWIN_ENV=development
export LOG_LEVEL=INFO
export PULSAR_URL=pulsar://localhost:6650

echo "📝 Configuration:"
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   Environment: $DARWIN_ENV"
echo ""

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn not found. Installing..."
    pip install uvicorn[standard]
fi

echo "🔧 Starting FastAPI server..."
echo ""

# Run uvicorn
uvicorn app.main:app \
    --host $HOST \
    --port $PORT \
    --reload \
    --log-level info

# Cleanup on exit
trap "echo ''; echo '🛑 Darwin Core stopped'; exit 0" INT TERM



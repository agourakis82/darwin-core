#!/bin/bash
# Generate Python code from Protocol Buffers

set -e

echo "=================================="
echo "Generating Python from Protos"
echo "=================================="
echo ""

# Check if grpcio-tools is installed
if ! python3 -c "import grpc_tools" 2>/dev/null; then
    echo "❌ grpcio-tools not found. Installing..."
    pip install grpcio-tools
fi

# Create output directory
mkdir -p app/protos

# Generate
echo "📝 Generating from protos..."

python3 -m grpc_tools.protoc \
    -I./protos \
    --python_out=./app/protos \
    --grpc_python_out=./app/protos \
    --pyi_out=./app/protos \
    protos/kec/v1/kec.proto \
    protos/plugin/v1/plugin.proto \
    protos/events/v1/events.proto

echo "✅ Proto generation complete!"
echo ""
echo "Generated files:"
ls -lh app/protos/**/*.py | awk '{print "  - " $9}'
echo ""
echo "You can now import:"
echo "  from app.protos.kec.v1 import kec_pb2, kec_pb2_grpc"
echo "  from app.protos.plugin.v1 import plugin_pb2, plugin_pb2_grpc"
echo "  from app.protos.events.v1 import events_pb2"


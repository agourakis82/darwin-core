#!/bin/bash
# Setup Cloudflare Tunnel for DARWIN Core

set -e

echo "=========================================="
echo "Cloudflare Tunnel Setup for DARWIN Core"
echo "=========================================="
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "📥 Installing cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared-linux-amd64
    sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
    echo "✅ cloudflared installed"
else
    echo "✅ cloudflared already installed"
fi

echo ""
echo "🔐 Step 1: Login to Cloudflare"
echo "   This will open a browser window"
cloudflared tunnel login

echo ""
echo "🌐 Step 2: Creating tunnel 'darwin-core'"
TUNNEL_OUTPUT=$(cloudflared tunnel create darwin-core 2>&1)
TUNNEL_ID=$(echo "$TUNNEL_OUTPUT" | grep -oP 'Created tunnel darwin-core with id \K[a-f0-9-]+')

if [ -z "$TUNNEL_ID" ]; then
    echo "⚠️  Tunnel may already exist, trying to get existing ID..."
    TUNNEL_ID=$(cloudflared tunnel list | grep darwin-core | awk '{print $1}')
fi

echo "✅ Tunnel ID: $TUNNEL_ID"

echo ""
echo "🎫 Step 3: Getting tunnel token"
TOKEN=$(cloudflared tunnel token $TUNNEL_ID 2>&1 | grep -oP 'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+')

if [ -z "$TOKEN" ]; then
    echo "❌ Failed to get token. Run manually:"
    echo "   cloudflared tunnel token darwin-core"
    exit 1
fi

echo "✅ Token obtained"

echo ""
echo "📝 Step 4: Creating Kubernetes secret"
kubectl create secret generic cloudflared-secret \
  --from-literal=token="$TOKEN" \
  -n darwin \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret created in K8s"

echo ""
echo "🚀 Step 5: Deploying cloudflared to K8s"
kubectl apply -f k8s/cloudflared-deployment.yaml

echo "✅ Cloudflared deployed"

echo ""
echo "🌍 Step 6: Configure DNS"
echo "   Run this command OR do it in Cloudflare dashboard:"
echo ""
echo "   cloudflared tunnel route dns darwin-core mcp-public.agourakis.med.br"
echo ""
read -p "Press Enter after configuring DNS..."

cloudflared tunnel route dns darwin-core mcp-public.agourakis.med.br || true

echo ""
echo "=========================================="
echo "✅ Cloudflare Tunnel Setup Complete!"
echo "=========================================="
echo ""
echo "Your DARWIN Core is now accessible at:"
echo "  https://mcp-public.agourakis.med.br"
echo ""
echo "Test it:"
echo "  curl https://mcp-public.agourakis.med.br/api/v1/health"
echo ""
echo "Configure MCP (Claude Desktop):"
echo "  URL: https://mcp-public.agourakis.med.br/api/v1/mcp"
echo ""
echo "Configure Custom GPT:"
echo "  OpenAPI: https://mcp-public.agourakis.med.br/openapi.json"
echo ""


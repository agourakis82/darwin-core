#!/bin/bash
# DARWIN Core 2.0 - Deploy Script
# Deploys complete modular architecture to Kubernetes

set -e

echo "========================================"
echo "DARWIN Core 2.0 - Kubernetes Deployment"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ docker not found. Please install Docker first.${NC}"
    exit 1
fi

echo -e "${YELLOW}📦 Step 1: Creating namespaces${NC}"
kubectl apply -f k8s/namespace.yaml
echo ""

echo -e "${YELLOW}🔧 Step 2: Installing NVIDIA Device Plugin${NC}"
kubectl apply -f k8s/nvidia-device-plugin.yaml
echo ""

echo -e "${YELLOW}🗄️ Step 3: Deploying ChromaDB${NC}"
kubectl apply -f k8s/chromadb-statefulset.yaml
echo ""

echo -e "${YELLOW}📡 Step 4: Deploying Apache Pulsar${NC}"
kubectl apply -f k8s/pulsar-cluster.yaml
echo ""

echo -e "${YELLOW}📊 Step 5: Deploying Monitoring Stack${NC}"
kubectl apply -f k8s/monitoring-stack.yaml
echo ""

echo -e "${YELLOW}🧠 Step 6: Building Darwin Core image${NC}"
docker build -t darwin-core:2.0.0 -f Dockerfile .
echo ""

echo -e "${YELLOW}🚀 Step 7: Deploying Darwin Core${NC}"
kubectl apply -f k8s/core-deployment.yaml
echo ""

echo -e "${YELLOW}🌐 Step 8: Deploying Envoy Gateway (HTTP/3)${NC}"
kubectl apply -f k8s/envoy-gateway-http3.yaml
echo ""

echo -e "${YELLOW}☁️  Step 8.5: Deploy Cloudflared (optional - for MCP/Custom GPT)${NC}"
echo "   To enable external access via Cloudflare Tunnel:"
echo "   1. Create tunnel: cloudflared tunnel create darwin-core"
echo "   2. Get token: cloudflared tunnel token darwin-core"
echo "   3. Update secret: kubectl create secret generic cloudflared-secret --from-literal=token='YOUR_TOKEN' -n darwin"
echo "   4. Deploy: kubectl apply -f k8s/cloudflared-deployment.yaml"
echo "   Skipping for now (manual setup required)"
echo ""

echo -e "${YELLOW}🧬 Step 9: Building Biomaterials Plugin image${NC}"
cd ../darwin-plugin-biomaterials
docker build -t darwin-plugin-biomaterials:2.0.0 -f Dockerfile .
cd ../darwin-core
echo ""

echo -e "${YELLOW}🔬 Step 10: Deploying Biomaterials Plugin${NC}"
kubectl apply -f ../darwin-plugin-biomaterials/k8s/deployment.yaml
echo ""

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "==================== STATUS ===================="
echo ""

# Wait for pods to be ready
echo -e "${YELLOW}⏳ Waiting for pods to be ready (may take 2-3 minutes)...${NC}"
kubectl wait --for=condition=ready pod -l app=darwin-core -n darwin --timeout=180s
kubectl wait --for=condition=ready pod -l app=darwin-chromadb -n darwin --timeout=180s

echo ""
echo -e "${GREEN}📊 Pod Status:${NC}"
kubectl get pods -n darwin
echo ""

echo -e "${GREEN}🌐 Services:${NC}"
kubectl get svc -n darwin
echo ""

echo -e "${GREEN}📈 Monitoring:${NC}"
kubectl get svc -n darwin-monitoring
echo ""

echo "==================== ENDPOINTS ===================="
echo ""
echo -e "${GREEN}DARWIN Core:${NC}"
echo "  HTTP/2: http://$(kubectl get svc darwin-core -n darwin -o jsonpath='{.spec.clusterIP}'):8090"
echo "  gRPC:   $(kubectl get svc darwin-core -n darwin -o jsonpath='{.spec.clusterIP}'):50051"
echo ""

echo -e "${GREEN}Envoy Gateway (HTTP/3):${NC}"
ENVOY_IP=$(kubectl get svc envoy-gateway -n darwin -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
if [ "$ENVOY_IP" != "pending" ]; then
    echo "  HTTP/3: https://${ENVOY_IP}"
    echo "  HTTP/2: http://${ENVOY_IP}"
else
    echo "  LoadBalancer IP: ${ENVOY_IP}"
fi
echo ""

echo -e "${GREEN}Grafana:${NC}"
GRAFANA_IP=$(kubectl get svc grafana -n darwin-monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
echo "  http://${GRAFANA_IP}:3000 (admin/darwin2025)"
echo ""

echo -e "${GREEN}Jaeger:${NC}"
JAEGER_IP=$(kubectl get svc jaeger-query -n darwin-monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
echo "  http://${JAEGER_IP}:16686"
echo ""

echo "==================== NEXT STEPS ===================="
echo ""
echo "1. Test Core health:"
echo "   kubectl port-forward -n darwin svc/darwin-core 8090:8090"
echo "   curl http://localhost:8090/api/v1/health"
echo ""
echo "2. Test gRPC communication:"
echo "   kubectl port-forward -n darwin svc/darwin-core 50051:50051"
echo "   grpcurl -plaintext localhost:50051 list"
echo ""
echo "3. View logs:"
echo "   kubectl logs -f -n darwin -l app=darwin-core"
echo "   kubectl logs -f -n darwin -l app=darwin-plugin-biomaterials"
echo ""
echo "4. Access Grafana:"
echo "   kubectl port-forward -n darwin-monitoring svc/grafana 3000:3000"
echo "   Open: http://localhost:3000"
echo ""
echo "5. Access Jaeger:"
echo "   kubectl port-forward -n darwin-monitoring svc/jaeger-query 16686:16686"
echo "   Open: http://localhost:16686"
echo ""
echo -e "${GREEN}🎉 DARWIN 2.0 is ready!${NC}"


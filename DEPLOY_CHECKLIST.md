# DARWIN 2.0 - Deploy Checklist

**Data**: 14 de Outubro de 2025  
**Versão**: 2.0.0

---

## ✅ Pré-Requisitos

### Sistema

- [ ] Kubernetes 1.28+ instalado
- [ ] kubectl configurado
- [ ] Docker instalado e rodando
- [ ] Acesso ao registry de imagens (ou build local)
- [ ] 192GB RAM disponível
- [ ] 20GB VRAM (1 GPU NVIDIA)

### NVIDIA GPU Support

- [ ] NVIDIA drivers instalados no node
- [ ] nvidia-docker2 configurado
- [ ] Kubernetes NVIDIA device plugin (será instalado pelo script)

---

## ✅ Componentes Implementados

### Darwin Core

- [x] `app/main.py` - FastAPI + gRPC + Pulsar + Agentic
- [x] `app/services/semantic_memory.py` - Memória semântica
- [x] `app/services/chroma_client.py` - ChromaDB client
- [x] `app/services/continuous_learning.py` - ML/RL automático
- [x] `app/services/pulsar_client.py` - Apache Pulsar (asyncio bridge)
- [x] `app/services/plugin_grpc_client.py` - gRPC client
- [x] `app/services/agentic_orchestrator.py` - Self-healing
- [x] `app/grpc_services/core_server.py` - gRPC server
- [x] `app/routers/memory_rest.py` - REST API integrado
- [x] `app/routers/mcp.py` - MCP server
- [x] `app/routers/health.py` - Health checks
- [x] `app/multi_ai/` - Multi-AI debate (copied)
- [x] `app/config/settings.py` - Configurações
- [x] Protocol Buffers (kec, plugin, events)
- [x] Dockerfile multi-stage
- [x] requirements-core.txt

### Plugin Biomaterials

- [x] `app/main.py` - FastAPI + gRPC
- [x] `app/grpc_service.py` - KECService + PluginService
- [x] `app/services/` - KEC Calculator, MicroCT, GNN
- [x] Protocol Buffers (copied)
- [x] Dockerfile CUDA
- [x] requirements.txt

### Kubernetes Manifests

- [x] `k8s/namespace.yaml` - Namespaces
- [x] `k8s/core-deployment.yaml` - Core + RBAC
- [x] `k8s/chromadb-statefulset.yaml` - ChromaDB
- [x] `k8s/pulsar-cluster.yaml` - Pulsar
- [x] `k8s/envoy-gateway-http3.yaml` - HTTP/3 gateway
- [x] `k8s/monitoring-stack.yaml` - Prometheus + Grafana + Jaeger
- [x] `k8s/nvidia-device-plugin.yaml` - GPU support
- [x] `darwin-plugin-biomaterials/k8s/deployment.yaml` - Plugin

### Scripts e Testes

- [x] `deploy.sh` - Deploy automatizado
- [x] `generate_protos.sh` - Gerar Python de protos
- [x] `test_components.py` - Teste de componentes
- [x] `test_grpc.py` - Teste gRPC

### Documentação

- [x] `README.md` (Core)
- [x] `darwin-plugin-biomaterials/README.md`
- [x] `docs/DARWIN_MODULAR_ARCHITECTURE_2025.md`
- [x] `docs/PLUGIN_DEVELOPMENT_GUIDE.md`
- [x] `DARWIN_2.0_IMPLEMENTATION_SUMMARY.md`

---

## ✅ Testes de Componentes

### Resultados do Test

```
✅ PASS pulsar (asyncio bridge funciona)
✅ PASS grpc_server (gRPC server inicia)
⚠️  SKIP agentic (K8s não configurado localmente - OK)
✅ PASS semantic_memory (ChromaDB conecta)
✅ PASS continuous_learning (ML engine funciona)
✅ PASS plugin_client (gRPC client criado)

Total: 5/6 passed (1 skip esperado)
```

**Status**: ✅ Componentes funcionando!

---

## ✅ Pré-Deploy Steps

### 1. Gerar Protocol Buffers

```bash
cd darwin-core
./generate_protos.sh
```

**Resultado esperado**:
```
✅ Proto generation complete!
Generated files:
  - app/protos/kec/v1/kec_pb2.py
  - app/protos/kec/v1/kec_pb2_grpc.py
  - app/protos/plugin/v1/plugin_pb2.py
  - app/protos/plugin/v1/plugin_pb2_grpc.py
  - app/protos/events/v1/events_pb2.py
```

### 2. Gerar Certificado TLS (para Envoy HTTP/3)

```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout /tmp/tls.key \
  -out /tmp/tls.crt \
  -days 365 -nodes \
  -subj "/CN=darwin-core.local"

# Criar secret no K8s (após criar namespace)
kubectl create namespace darwin
kubectl create secret tls envoy-tls-cert \
  --cert=/tmp/tls.crt \
  --key=/tmp/tls.key \
  -n darwin
```

### 3. Build Images (Opcional se usar registry)

```bash
# Build Core
cd darwin-core
docker build -t darwin-core:2.0.0 .

# Build Plugin Biomaterials
cd ../darwin-plugin-biomaterials  
docker build -t darwin-plugin-biomaterials:2.0.0 .
```

---

## 🚀 Deploy

### Executar Deploy Script

```bash
cd darwin-core
./deploy.sh
```

### O que o script faz:

1. ✅ Cria namespaces (`darwin`, `darwin-monitoring`, `pulsar`)
2. ✅ Instala NVIDIA device plugin
3. ✅ Deploy ChromaDB (StatefulSet + PV)
4. ✅ Deploy Apache Pulsar
5. ✅ Deploy Monitoring (Prometheus + Grafana + Jaeger)
6. ✅ Build Core image
7. ✅ Deploy Core (Deployment + Service + RBAC)
8. ✅ Deploy Envoy Gateway (HTTP/3)
9. ✅ Build Plugin Biomaterials
10. ✅ Deploy Plugin Biomaterials (com GPU)

**Tempo estimado**: 5-10 minutos (primeiro deploy)

---

## ✅ Validação Pós-Deploy

### 1. Verificar Pods

```bash
kubectl get pods -n darwin
```

**Esperado**:
```
NAME                                         READY   STATUS    RESTARTS
darwin-core-xxx                              1/1     Running   0
darwin-core-yyy                              1/1     Running   0
darwin-chromadb-0                            1/1     Running   0
darwin-plugin-biomaterials-xxx               1/1     Running   0
envoy-gateway-xxx                            1/1     Running   0
```

### 2. Verificar Services

```bash
kubectl get svc -n darwin
```

**Esperado**:
```
NAME                          TYPE           PORT(S)
darwin-core                   ClusterIP      8090/TCP, 50051/TCP
darwin-chromadb               ClusterIP      8000/TCP
darwin-plugin-biomaterials    ClusterIP      8001/TCP, 50052/TCP
envoy-gateway                 LoadBalancer   443/UDP, 80/TCP
```

### 3. Verificar Logs

```bash
# Core
kubectl logs -n darwin -l app=darwin-core --tail=50

# Plugin
kubectl logs -n darwin -l app=darwin-plugin-biomaterials --tail=50
```

**Esperado no Core**:
```
✅ DARWIN Core 2.0 - Ready
   Pulsar: True
   gRPC: True
   Agentic: True
   Routers: True
```

### 4. Test Health Endpoint

```bash
kubectl port-forward -n darwin svc/darwin-core 8090:8090 &
sleep 2
curl http://localhost:8090/api/v1/health | jq
```

**Esperado**:
```json
{
  "status": "healthy",
  "service": "DARWIN Core 2.0",
  "version": "2.0.0",
  "components": {
    "fastapi": "operational",
    "pulsar": "operational",
    "grpc": "operational",
    "agentic": "operational"
  }
}
```

### 5. Test gRPC Communication

```bash
kubectl port-forward -n darwin svc/darwin-core 50051:50051 &
kubectl port-forward -n darwin svc/darwin-plugin-biomaterials 50052:50052 &
sleep 2

cd darwin-core
python3 test_grpc.py
```

**Esperado**:
```
✅ Health Check: PASS
✅ Metadata: PASS
✅ KEC Analysis: PASS
✅ Streaming: PASS

🎉 All tests passed!
```

### 6. Test Memory Save/Search

```bash
# Save
curl -X POST http://localhost:8090/api/v1/memory/save \
  -H "Content-Type: application/json" \
  -d '{
    "title": "DARWIN 2.0 First Memory",
    "content": "Sistema modular deployado com sucesso!",
    "domain": "research",
    "platform": "claude-code",
    "tags": ["deploy", "success"]
  }'

# Search
curl -X POST http://localhost:8090/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deploy sucesso",
    "top_k": 5
  }'
```

### 7. Access Monitoring

```bash
# Grafana
kubectl port-forward -n darwin-monitoring svc/grafana 3000:3000
# http://localhost:3000 (admin/darwin2025)

# Jaeger
kubectl port-forward -n darwin-monitoring svc/jaeger-query 16686:16686
# http://localhost:16686
```

---

## 🔧 Troubleshooting

### Pods não iniciam

```bash
# Ver eventos
kubectl describe pod -n darwin darwin-core-xxx

# Ver logs detalhados
kubectl logs -n darwin darwin-core-xxx --previous
```

### GPU não alocada

```bash
# Verificar NVIDIA device plugin
kubectl get pods -n kube-system | grep nvidia

# Ver GPUs disponíveis
kubectl describe node | grep -A 10 "Allocatable"
```

### Pulsar connection failed

```bash
# Ver logs Pulsar
kubectl logs -n pulsar darwin-pulsar-0

# Restart Pulsar
kubectl delete pod -n pulsar darwin-pulsar-0
```

### ChromaDB não conecta

```bash
# Ver logs
kubectl logs -n darwin darwin-chromadb-0

# Test conexão
kubectl port-forward -n darwin darwin-chromadb-0 8000:8000
curl http://localhost:8000/api/v1/heartbeat
```

---

## 📊 Métricas de Sucesso

### Performance

- [ ] Latência p95 < 100ms
- [ ] Throughput > 100 req/s
- [ ] GPU utilization 70-90% durante análise
- [ ] Memory usage < 80%

### Funcionalidade

- [ ] Save memory funciona
- [ ] Search memory retorna resultados
- [ ] gRPC Core → Plugin funciona
- [ ] KEC analysis retorna métricas corretas
- [ ] Continuous learning registra interações

### Observability

- [ ] Grafana dashboards acessíveis
- [ ] Jaeger mostra traces
- [ ] Prometheus scraping métricas
- [ ] Logs estruturados visíveis

### Resiliência

- [ ] Pods restart após falha
- [ ] Health checks funcionam
- [ ] Circuit breakers ativam
- [ ] Agentic layer detecta falhas

---

## 🎯 Próximos Passos Pós-Deploy

1. **Validar KEC com dados reais**
   - Upload MicroCT
   - Análise completa
   - Validar H/κ/σ/ϕ/d_perc

2. **Treinar modelos ML**
   - Continuous learning com 50+ interações
   - Validar personalização

3. **Deploy plugin Chemistry**
   - Quando tiver demanda
   - Usar template de plugin

4. **Expandir para multi-node**
   - Quando node 2 chegar
   - Distribuir plugins
   - Pulsar cluster

---

## 🎉 Sistema Pronto!

Se todos os itens acima passarem:

**DARWIN 2.0 está production-ready!** ✅

- Arquitetura modular ✅
- gRPC + Pulsar + HTTP/3 ✅
- AI Agentic ✅
- Observability ✅
- GPU support ✅
- Documentação completa ✅

**Pode fazer deploy em produção!** 🚀


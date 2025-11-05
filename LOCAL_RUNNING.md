# 🚀 Darwin Core Rodando Localmente!

**Status**: ✅ ATIVO  
**URL**: http://localhost:8090

---

## 📊 Informações do Servidor

| Item | Valor |
|------|-------|
| **Host** | 0.0.0.0 (todas interfaces) |
| **Port** | 8090 |
| **Log File** | `darwin-local.log` |
| **Reload** | ✅ Ativado (detecta mudanças no código) |
| **Environment** | Development |

---

## 🧪 Endpoints Disponíveis

### Health Checks
```bash
# Principal
curl http://localhost:8090/

# MCP Health
curl http://localhost:8090/mcp/health
```

### MCP Tools
```bash
# Lista de tools
curl http://localhost:8090/mcp/tools

# Executar tool
curl -X POST http://localhost:8090/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "darwin_rag_query",
    "arguments": {
      "query": "biomaterials scaffolds",
      "top_k": 5
    }
  }'
```

### Custom GPT API
```bash
# Analyze (requer autenticação)
curl -X POST http://localhost:8090/api/v1/analyze \
  -H "Authorization: Bearer darwin_local_dev_token_2025" \
  -H "Content-Type: application/json" \
  -d '{"data": "test", "analysis_type": "biomaterials"}'
```

### Documentação Interativa
- **Swagger UI**: http://localhost:8090/docs
- **ReDoc**: http://localhost:8090/redoc
- **OpenAPI JSON**: http://localhost:8090/openapi.json

---

## 📋 Gerenciamento

### Ver Logs em Tempo Real
```bash
cd /home/agourakis82/workspace/kec-biomaterials-scaffolds/darwin-core
tail -f darwin-local.log
```

### Ver Processo Rodando
```bash
ps aux | grep uvicorn | grep 8090
```

### Parar o Servidor
```bash
# Encontrar PID
ps aux | grep uvicorn | grep 8090 | awk '{print $2}'

# Parar (substituir <PID> pelo número encontrado)
kill <PID>

# Ou parar todos os uvicorn na porta 8090
pkill -f "uvicorn.*8090"
```

### Reiniciar
```bash
cd /home/agourakis82/workspace/kec-biomaterials-scaffolds/darwin-core

# Parar
pkill -f "uvicorn.*8090"

# Iniciar novamente
./run-local.sh > darwin-local.log 2>&1 &
```

---

## 🔧 Configuração

As variáveis de ambiente estão definidas em `run-local.sh`:

```bash
HOST=0.0.0.0
PORT=8090
DARWIN_API_TOKEN=darwin_local_dev_token_2025
DARWIN_ENV=development
LOG_LEVEL=INFO

# Services (opcionais - graceful degradation se não disponíveis)
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379
OLLAMA_URL=http://localhost:11434
PULSAR_URL=pulsar://localhost:6650
```

---

## 🧰 Desenvolvimento

### Hot Reload
O servidor está configurado com `--reload`, então mudanças no código são detectadas automaticamente!

### Testar Mudanças
1. Edite o código em `app/`
2. Salve o arquivo
3. Uvicorn recarrega automaticamente
4. Teste com `curl` ou no browser

### Debug
```bash
# Ver últimas 100 linhas do log
tail -100 darwin-local.log

# Filtrar por erros
grep -i error darwin-local.log

# Filtrar por warnings
grep -i warning darwin-local.log
```

---

## 🌐 Acesso Externo

### Cursor AI
O Darwin Core está acessível via:
- **Local**: http://localhost:8090
- **Público (K8s)**: https://gpt.agourakis.med.br

### Navegador
Abra: http://localhost:8090/docs

### Postman/Insomnia
Base URL: `http://localhost:8090`

---

## 📊 Status Atual

```bash
# Check rápido
curl -s http://localhost:8090/ | jq .

# MCP Tools count
curl -s http://localhost:8090/mcp/tools | jq '.tools | length'

# Memory check
ps aux | grep "uvicorn.*8090" | awk '{print $11}'
```

---

## 🎯 Features Ativas

- ✅ FastAPI REST API
- ✅ MCP Server (6 tools)
- ✅ Custom GPT API endpoints
- ✅ OpenAPI documentation
- ✅ Hot reload (desenvolvimento)
- ✅ CORS habilitado
- ✅ OpenTelemetry (tentará conectar ao Jaeger)
- ✅ gRPC server (porta 50051)
- ✅ Auto-training pipeline
- ✅ Agentic orchestrator (com K8s local)

---

## ⚠️ Notas

1. **Dependências Externas**: Qdrant, Redis, Ollama, Pulsar são opcionais. O sistema degrada graciosamente se não estiverem disponíveis.

2. **Performance**: Esta é uma instância de desenvolvimento. Para produção, use a versão no K8s.

3. **Porta 8090**: Se já estiver em uso, edite `PORT` no `run-local.sh`.

4. **Auto-reload**: Útil para desenvolvimento, mas consome mais recursos.

---

## 🚀 Próximos Passos

### Integrar com IDE
Configure seu IDE/Cursor para:
- Breakpoints: Use o debugger Python
- Tests: Execute testes com pytest
- Linting: Use ruff ou pylint

### Testar MCP Tools
```bash
# RAG Query
curl -X POST http://localhost:8090/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "darwin_rag_query",
    "arguments": {"query": "titanium scaffolds"}
  }'

# Save Memory
curl -X POST http://localhost:8090/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "darwin_save_memory",
    "arguments": {
      "content": "Titanium has excellent biocompatibility",
      "title": "Titanium Biocompatibility",
      "domain": "biomaterials"
    }
  }'
```

---

## 📖 Documentação Completa

- **MCP Integration**: `docs/MCP_INTEGRATION.md`
- **Custom GPT API**: `docs/CUSTOM_GPT_API.md`
- **Deploy Status**: `SUCESSO_DEPLOY_2025.md`
- **K8s Deployment**: `README_K8S_DEPLOY.md`

---

**Darwin Core 2025.1.0 rodando localmente no Cursor!** 🎉

**PID do processo**: Veja com `ps aux | grep "uvicorn.*8090"`  
**Log file**: `darwin-local.log`  
**Última atualização**: 27 de Outubro de 2025



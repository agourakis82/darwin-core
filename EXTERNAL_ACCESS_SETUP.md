# DARWIN 2.0 - Configuração de Acesso Externo (MCP + Custom GPT)

**Data**: 14 de Outubro de 2025  
**Objetivo**: Expor DARWIN Core para MCP (Claude) e Custom GPT

---

## Opções de Exposição

### Opção 1: Cloudflare Tunnel (Recomendado)

**Vantagens**:
- ✅ Gratuito
- ✅ HTTPS automático
- ✅ Sem necessidade de IP público
- ✅ HTTP/3 + QUIC suportado
- ✅ DDoS protection

**Configuração**:

```bash
# 1. Instalar cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# 2. Login no Cloudflare
cloudflared tunnel login

# 3. Criar tunnel
cloudflared tunnel create darwin-core

# 4. Configurar tunnel
cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: <TUNNEL_ID>
credentials-file: /home/agourakis82/.cloudflared/<TUNNEL_ID>.json

ingress:
  # MCP endpoint
  - hostname: mcp-public.agourakis.med.br
    service: http://localhost:8090
    originRequest:
      noTLSVerify: true
      http2Origin: true
  
  # Fallback
  - service: http_status:404
EOF

# 5. Criar DNS record no Cloudflare
cloudflared tunnel route dns darwin-core mcp-public.agourakis.med.br

# 6. Run tunnel
cloudflared tunnel run darwin-core
```

**Para Kubernetes** (melhor):

```yaml
# k8s/cloudflared-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cloudflared
  namespace: darwin
spec:
  replicas: 2  # HA
  selector:
    matchLabels:
      app: cloudflared
  template:
    metadata:
      labels:
        app: cloudflared
    spec:
      containers:
      - name: cloudflared
        image: cloudflare/cloudflared:latest
        args:
        - tunnel
        - --no-autoupdate
        - run
        - --token
        - $(TUNNEL_TOKEN)
        env:
        - name: TUNNEL_TOKEN
          valueFrom:
            secretKeyRef:
              name: cloudflared-secret
              key: token
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"

---
apiVersion: v1
kind: Secret
metadata:
  name: cloudflared-secret
  namespace: darwin
type: Opaque
stringData:
  token: "YOUR_TUNNEL_TOKEN_HERE"
```

### Opção 2: Nginx Reverse Proxy (Tradicional)

**Configuração**:

```nginx
# /etc/nginx/sites-available/darwin-core
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name mcp-public.agourakis.med.br;

    # SSL certificates (Let's Encrypt)
    ssl_certificate /etc/letsencrypt/live/mcp-public.agourakis.med.br/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp-public.agourakis.med.br/privkey.pem;
    
    # SSL config
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Proxy to Darwin Core
    location / {
        proxy_pass http://localhost:8090;
        proxy_http_version 1.1;
        
        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # CORS (for Custom GPT)
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization';
    }
}
```

---

## MCP Configuration (Claude Desktop)

### Arquivo: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "darwin-memory-production": {
      "command": "uvx",
      "args": [
        "mcp-server-fetch",
        "https://mcp-public.agourakis.med.br/api/v1/mcp"
      ],
      "env": {
        "DARWIN_TOKEN": "darwin_MCP_2025_PERMANENT_TOKEN_FOR_CLAUDE"
      }
    },
    "darwin-filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "/home/agourakis82/workspace/kec-biomaterials-scaffolds"
      ]
    }
  }
}
```

**Testar**:
```bash
# Via Claude Desktop, executar:
"Salve uma memória no DARWIN: teste de deploy 2.0"
```

---

## Custom GPT Configuration

### OpenAPI Schema URL

**URL pública**: `https://mcp-public.agourakis.med.br/openapi.json`

### Custom GPT Setup

1. **Ir para**: https://chat.openai.com/gpts/editor
2. **Configure** → **Actions**
3. **Import from URL**: `https://mcp-public.agourakis.med.br/openapi.json`
4. **Authentication**: None (ou API Key se implementar)
5. **Privacy**: Use your data to improve model (OFF)

### Testar Custom GPT

```
"Salve no DARWIN: teste de deploy kubernetes"
```

**Esperado**: 
```
✅ Memory saved: mem_abc123
```

---

## Envoy Gateway Update (para MCP/Custom GPT)

### Adicionar rotas MCP ao Envoy

```yaml
# Atualizar k8s/envoy-gateway-http3.yaml

routes:
# MCP endpoints (Custom GPT compatibility)
- match:
    prefix: "/api/v1/mcp/"
  route:
    cluster: darwin_core
    timeout: 30s
    retry_policy:
      retry_on: 5xx
      num_retries: 3

# Legacy MCP aliases
- match:
    prefix: "/darwinSaveMemory"
  route:
    cluster: darwin_core
    prefix_rewrite: "/api/v1/mcp/darwinSaveMemory"

- match:
    prefix: "/darwinSearchMemory"
  route:
    cluster: darwin_core
    prefix_rewrite: "/api/v1/mcp/darwinSearchMemory"

# OpenAPI schema
- match:
    prefix: "/openapi.json"
  route:
    cluster: darwin_core
```

---

## Checklist Pré-Deploy

### DNS e Certificados

- [ ] Domínio configurado: `mcp-public.agourakis.med.br`
- [ ] DNS apontando para LoadBalancer IP
- [ ] Certificado SSL (Let's Encrypt ou Cloudflare)

### Cloudflare Tunnel (se usar)

- [ ] Tunnel criado: `cloudflared tunnel create darwin-core`
- [ ] Token salvo no Secret K8s
- [ ] DNS route configurada
- [ ] Deployment do cloudflared no K8s

### Nginx (alternativa)

- [ ] Nginx instalado
- [ ] Certificado Let's Encrypt
- [ ] Config em `/etc/nginx/sites-available/`
- [ ] Port forwarding: 443 → 8090

### Testes de Acesso

- [ ] `curl https://mcp-public.agourakis.med.br/api/v1/health`
- [ ] `curl https://mcp-public.agourakis.med.br/openapi.json`
- [ ] MCP via Claude Desktop funciona
- [ ] Custom GPT consegue chamar endpoints

---

## Quick Setup (Cloudflare Tunnel Recomendado)

### Passo a Passo Rápido

```bash
# 1. Instalar cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

# 2. Login e criar tunnel
cloudflared tunnel login
cloudflared tunnel create darwin-core

# 3. Pegar token
cloudflared tunnel token darwin-core
# Copie o token que aparecer

# 4. Criar secret no K8s
kubectl create secret generic cloudflared-secret \
  --from-literal=token='YOUR_TOKEN_HERE' \
  -n darwin

# 5. Deploy cloudflared
kubectl apply -f k8s/cloudflared-deployment.yaml

# 6. Configurar DNS no Cloudflare dashboard
# Adicionar CNAME: mcp-public -> <TUNNEL_ID>.cfargotunnel.com

# 7. Testar
curl https://mcp-public.agourakis.med.br/api/v1/health
```

**Tempo**: ~15 minutos

---

## Diagrama de Acesso

```
┌─────────────────────────────────────────────────────────┐
│              EXTERNAL CLIENTS                            │
│                                                          │
│  Claude Desktop MCP    Custom GPT    HTTP/3 Clients      │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│          CLOUDFLARE TUNNEL / NGINX                       │
│                                                          │
│  https://mcp-public.agourakis.med.br                    │
│  ├─ TLS termination                                     │
│  ├─ DDoS protection                                     │
│  └─ HTTP/3 → HTTP/2                                     │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│            ENVOY GATEWAY (K8s)                          │
│                                                          │
│  ├─ Load balancing                                      │
│  ├─ Circuit breaking                                    │
│  ├─ Rate limiting                                       │
│  └─ Routing /api/v1/* → darwin-core                     │
└─────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────┐
│            DARWIN CORE (K8s Pods)                       │
│                                                          │
│  FastAPI REST + gRPC + Pulsar + Agentic                 │
└─────────────────────────────────────────────────────────┘
```

---

## Security

### Rate Limiting (Envoy)

```yaml
# Adicionar ao envoy-config
http_filters:
- name: envoy.filters.http.local_ratelimit
  typed_config:
    "@type": type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
    stat_prefix: http_local_rate_limiter
    token_bucket:
      max_tokens: 1000
      tokens_per_fill: 100
      fill_interval: 1s
```

### Authentication (opcional)

Para produção, considere adicionar:
- JWT tokens
- API keys
- OAuth2

---

## Custos

### Cloudflare Tunnel

- **Free tier**: Ilimitado (perfeito!)
- Bandwidth: Sem limite
- HTTP/3: Incluído

### Alternativas

- **Nginx**: $0 (self-hosted)
- **ngrok**: $8/mês (básico)
- **LoadBalancer K8s**: Depende do provider

---

## Conclusão

**Recomendação**: Use **Cloudflare Tunnel** - é:
- ✅ Gratuito
- ✅ Fácil de configurar
- ✅ HTTPS automático
- ✅ HTTP/3 + QUIC
- ✅ DDoS protection
- ✅ Sem necessidade de IP público

**Pronto para usar com MCP e Custom GPT após configurar tunnel!**


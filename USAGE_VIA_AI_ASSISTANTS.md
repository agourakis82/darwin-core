# DARWIN 2.0 - Uso via AI Assistants (Claude & Custom GPT)

**Data**: 14 de Outubro de 2025  
**Método Primário de Operação**: 🤖 Conversacional (Claude/Custom GPT)

---

## 🎯 Filosofia de Uso

**DARWIN 2.0 é operado CONVERSACIONALMENTE via AI assistants**:

✅ **Claude Desktop** (via MCP) - Método preferencial  
✅ **Custom GPT** (via OpenAPI Actions) - Alternativa  
✅ **Terminal/API** - Apenas para debug e admin  

**Você conversa naturalmente** - DARWIN executa ações automaticamente!

---

## 🤖 Configuração Claude Desktop (MCP)

### Arquivo: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "darwin-production": {
      "command": "uvx",
      "args": [
        "mcp-server-fetch",
        "https://mcp-public.agourakis.med.br/api/v1/mcp"
      ],
      "env": {
        "DARWIN_TOKEN": "darwin_MCP_2025_PERMANENT_TOKEN"
      }
    }
  }
}
```

### Testar no Claude

```
Você: "Salve no DARWIN: acabei de fazer deploy do DARWIN 2.0 com arquitetura modular"

Claude: ✅ Memory saved: mem_abc123

Você: "Busque no DARWIN tudo sobre deploy"

Claude: 📚 Found 3 results:
1. "Deploy do DARWIN 2.0" (há 2 minutos)
2. "Configuração Kubernetes" (ontem)
3. "Setup Cloudflare" (ontem)
```

---

## 💬 Configuração Custom GPT

### 1. Criar Custom GPT

**URL**: https://chat.openai.com/gpts/editor

### 2. Configurar Actions

**Import from URL**: `https://mcp-public.agourakis.med.br/openapi.json`

**Ou copiar schema** (se URL não funcionar):

```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "DARWIN Core 2.0",
    "version": "2.0.0"
  },
  "servers": [
    {"url": "https://mcp-public.agourakis.med.br"}
  ],
  "paths": {
    "/api/v1/mcp/darwinSaveMemory": {
      "post": {
        "summary": "Save memory",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "title": {"type": "string"},
                  "content": {"type": "string"},
                  "domain": {"type": "string"},
                  "platform": {"type": "string"},
                  "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["content"]
              }
            }
          }
        }
      }
    },
    "/api/v1/mcp/darwinSearchMemory": {
      "post": {
        "summary": "Search memory",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "query": {"type": "string"},
                  "top_k": {"type": "integer"},
                  "domain": {"type": "string"}
                },
                "required": ["query"]
              }
            }
          }
        }
      }
    }
  }
}
```

### 3. Testar Custom GPT

```
Você: "Salve no DARWIN: configurei Custom GPT com sucesso"

GPT: ✅ Salvei na sua memória DARWIN!

Você: "Busque o que eu disse sobre Custom GPT"

GPT: 📚 Encontrei:
"Configurei Custom GPT com sucesso" (agora mesmo)
```

---

## 📚 Casos de Uso via Conversação

### 1. Salvar Insights de Pesquisa

**Você (Claude/GPT)**:
```
"Salvekno DARWIN: descobri que scaffolds com porosidade bimodal 
(100μm + 300μm) têm melhor vascularização que unimodal. 
Testar com PCL na próxima rodada."
```

**Sistema**:
- ✅ Salva no ChromaDB (RAG++)
- ✅ Registra interação (Continuous Learning)
- ✅ Publica evento (Pulsar)
- ✅ Conta para treinamento automático

**Resultado**: Quando atingir 100 conversas sobre biomaterials, sistema treina modelo especializado automaticamente!

### 2. Buscar Conhecimento Prévio

**Você**:
```
"Busque no DARWIN tudo que eu já descobri sobre porosidade bimodal"
```

**Sistema**:
- ✅ Busca semântica (não precisa palavras exatas!)
- ✅ Personaliza resultados (sabe que biomaterials é sua expertise)
- ✅ Ranking inteligente (boost suas descobertas)

### 3. Ingerir Paper Científico

**Você**:
```
"Quero adicionar este paper ao DARWIN" 
[cola texto completo do paper]
```

**Claude/GPT**:
```
POST /api/v1/corpus/ingest/text
{
  "text": "...",
  "domain": "biomaterials",
  "title": "Scaffold Optimization Study",
  "tags": ["paper", "2024"]
}
```

**Sistema**:
- ✅ Chunka em ~50 partes (1000 chars cada)
- ✅ Salva tudo no RAG++
- ✅ Usa para treinar modelo
- ✅ Agora você pode fazer perguntas sobre o paper!

### 4. Treinar Modelo Manualmente

**Você**:
```
"Já tenho bastante conhecimento de biomaterials no DARWIN. 
Pode treinar um modelo especializado agora?"
```

**Claude/GPT**:
```
POST /api/v1/corpus/train-on-corpus?domain=biomaterials&force=true
```

**Sistema**:
- ✅ Coleta todas conversas + papers de biomaterials
- ✅ Fine-tune qwen2.5:32b
- ✅ Cria darwin-biomaterials-local-v2
- ✅ Auto-deploy (se melhor que v1)

### 5. Listar Modelos Disponíveis

**Você**:
```
"Quais modelos locais tenho disponíveis?"
```

**Claude/GPT**: 
```
GET /api/v1/models/list
```

**Retorna**:
```
📊 11 modelos locais:
- Llama 3.1 8B (geral)
- Qwen 2.5 32B (pesquisa avançada)
- Qwen 2.5 Coder 7B (código)
- LLaVA 13B (visão)
- DARWIN Biomaterials Expert (seu modelo!)
- DARWIN Medical Expert
- DARWIN Pharmacology Expert
- DARWIN Mathematics Expert
- DARWIN Quantum Expert
- DARWIN Philosophy Expert
- Nomic Embed (embeddings)
```

### 6. Adicionar Novo Modelo

**Você**:
```
"Acabei de instalar llama3.2:90b no Ollama. 
Pode adicionar ao DARWIN?"
```

**Claude/GPT**:
```
POST /api/v1/models/sync-ollama
```

**Sistema**:
- ✅ Detecta novo modelo
- ✅ Auto-registra no registry
- ✅ Disponível para uso imediatamente

---

## 🔄 Workflow Completo

### Dia 1-7: Coleta

```
Você conversa normalmente:
- Claude Code: "Analisei scaffold PCL..."
- ChatGPT: "Li paper sobre biocompatibilidade..."
- Gemini: "Dúvida sobre vascularização..."

Sistema:
→ Salva tudo no RAG++
→ Conta conversas por domínio
→ biomaterials: 45 conversas
```

### Dia 8: Ingestão de Papers

```
Você:
"Quero adicionar 10 papers sobre scaffolds ao DARWIN"

[cola textos ou upload PDFs]

Sistema:
→ Chunka 10 papers = ~500 chunks
→ biomaterials: 45 + 500 = 545 itens ✅
→ THRESHOLD ATINGIDO (100)!
```

### Dia 8: Treinamento Automático

```
Sistema (background):
→ Detecta threshold
→ Coleta 545 conversas + chunks
→ Fine-tune qwen2.5:32b
→ Cria darwin-biomaterials-local-v2
→ Testa modelo
→ Deploy automático ✅

Notificação:
"🎓 Novo modelo treinado: darwin-biomaterials-v2 
agora especializado em suas pesquisas!"
```

### Dia 9+: Uso do Modelo Especializado

```
Você (Claude):
"O que você sabe sobre porosidade bimodal?"

Sistema:
→ Usa darwin-biomaterials-v2 (SEU modelo!)
→ Resposta baseada em SUAS conversas + papers
→ Personalizado para VOCÊ!

Resposta é melhor porque:
✅ Treinou com seus dados
✅ Sabe seu contexto
✅ Usa sua terminologia
✅ Foca no que você pesquisa
```

---

## 🎯 Endpoints Essenciais (via Claude/GPT)

### Operação Diária

| Ação | Endpoint | Método |
|------|----------|--------|
| Salvar ideia | `/api/v1/mcp/darwinSaveMemory` | POST |
| Buscar conhecimento | `/api/v1/mcp/darwinSearchMemory` | POST |
| Ver perfil | `/api/v1/memory/profile` | GET |
| Ver modelos | `/api/v1/models/list` | GET |

### Ingestão de Conhecimento

| Ação | Endpoint | Método |
|------|----------|--------|
| Adicionar texto | `/api/v1/corpus/ingest/text` | POST |
| Upload PDF | `/api/v1/corpus/ingest/pdf` | POST |
| Batch upload | `/api/v1/corpus/ingest/batch` | POST |

### Treinamento

| Ação | Endpoint | Método |
|------|----------|--------|
| Status treino | `/api/v1/corpus/training-status` | GET |
| Treinar agora | `/api/v1/corpus/train-on-corpus` | POST |
| Adicionar modelo | `/api/v1/models/register` | POST |

---

## 🧪 Teste Antes do Deploy

```bash
# 1. Start Core
cd darwin-core
uvicorn app.main:app --host 0.0.0.0 --port 8090

# 2. Run MCP integration tests
python3 test_mcp_integration.py

# Esperado:
# ✅ MCP Save Memory: OK
# ✅ MCP Search Memory: OK
# ✅ Legacy Endpoints: OK
# ✅ OpenAPI Schema: OK
# ✅ Model Management: OK
# ✅ Corpus Ingestion: OK
# ✅ Training Status: OK
#
# 🎉 All MCP/Custom GPT endpoints working!
```

---

## 🚀 Pós-Deploy Checklist

### Claude Desktop

- [ ] Atualizar `~/.cursor/mcp.json` com URL production
- [ ] Testar: "Salve no DARWIN: teste production"
- [ ] Verificar: Salvou corretamente

### Custom GPT

- [ ] Importar schema: `https://mcp-public.agourakis.med.br/openapi.json`
- [ ] Testar: "Busque no DARWIN sobre deploy"
- [ ] Verificar: Retornou resultados

### Ingestão de Papers

- [ ] Upload primeiro paper (PDF)
- [ ] Verificar: Chunkado e salvo
- [ ] Buscar trecho do paper
- [ ] Verificar: RAG++ encontrou

### Auto-Training

- [ ] Aguardar 100 conversas OU
- [ ] Upload 50+ papers (atinge threshold)
- [ ] Verificar: Treinamento disparado
- [ ] Aguardar: ~30-60 min
- [ ] Verificar: Novo modelo disponível
- [ ] Testar: Modelo especializado funciona

---

## 💡 Exemplos de Conversação

### Pesquisa Diária

```
Você → Claude: "Analisei scaffold hoje. Porosidade 87%, 
KEC mostrou sigma 1.8 (small-world!). Biocompatibilidade 
excelente. Salve no DARWIN."

Claude → DARWIN: POST /api/v1/mcp/darwinSaveMemory
{
  "title": "Análise Scaffold Diária",
  "content": "...",
  "domain": "biomaterials",
  "tags": ["scaffold", "kec", "small-world"]
}

DARWIN: ✅ Salvo! Registrado para continuous learning.
```

### Revisão de Literatura

```
Você → Claude: "Estou lendo paper sobre Ollivier-Ricci 
curvature. [cola 20 páginas]. Adicione ao DARWIN para 
eu consultar depois."

Claude → DARWIN: POST /api/v1/corpus/ingest/text
{
  "text": "...",  // 20 páginas
  "domain": "mathematics",
  "title": "Ollivier-Ricci Curvature Paper",
  "tags": ["curvature", "mathematics", "paper"]
}

DARWIN: ✅ Processado 45 chunks. Adicionado ao RAG++.
Biomaterials está com 95 conversas (5 para treinar modelo!).
```

### Consulta Cross-Domain

```
Você → Claude: "Busque no DARWIN como curvatura de Ricci 
se relaciona com biomaterials"

Claude → DARWIN: POST /api/v1/memory/search
{
  "query": "curvatura Ricci biomaterials relação",
  "top_k": 10
}

DARWIN → Claude: 
📚 Encontrei 8 resultados relevantes:

1. "Análise KEC usa Forman-Ricci e Ollivier-Ricci para..."
2. "Paper sobre curvatura em scaffolds..."
3. "Sua conversa sobre matemática e biomaterials..."

[DARWIN detectou que você frequentemente conecta 
mathematics ↔ biomaterials, então boosted esses resultados!]
```

### Gerenciar Modelos

```
Você → Claude: "Quais modelos especializados tenho 
treinados no DARWIN?"

Claude → DARWIN: GET /api/v1/models/list?tags=darwin

DARWIN: 
🤖 6 modelos DARWIN especializados:
- darwin-biomaterials-local-v2 (treinado com 150 conversas)
- darwin-medical-local-v1 (85 conversas)
- darwin-pharmacology-local-v1 (60 conversas)
- darwin-mathematics-local-v1 (120 conversas)
- darwin-quantum-local-v1 (45 conversas)
- darwin-philosophy-local-v1 (70 conversas)

Você → Claude: "Use o modelo biomaterials v2 para 
analisar este scaffold"

Claude: [usa darwin-biomaterials-v2 via debate]
```

---

## 🎓 Treinamento Automático (Seu Segundo Cérebro)

### Como Funciona

```
Suas Conversas (todas plataformas)
    ↓
MCP salva no DARWIN
    ↓
Continuous Learning registra
    ↓
Counter: biomaterials = 100 conversas ✅
    ↓
Auto-Training Pipeline dispara
    ↓
Fine-tune qwen2.5:32b com suas conversas
    ↓
Cria darwin-biomaterials-v2
    ↓
Avalia modelo (perplexity)
    ↓
Deploy automático se melhor!
    ↓
Próximas conversas usam SEU modelo!
```

### Monitorar Treinamento

**Via Claude**:
```
"Qual status do treinamento automático de modelos?"

GET /api/v1/corpus/training-status

Retorno:
{
  "running": true,
  "domain_counts": {
    "biomaterials": 98,  // Quase lá!
    "mathematics": 45,
    "philosophy": 32
  },
  "next_training": {
    "biomaterials": "ready in 2 conversations"
  },
  "model_versions": {
    "biomaterials": 2,  // v2 já treinado
    "mathematics": 1
  }
}
```

---

## 📊 Dashboard de Status (via Conversação)

**Perguntar ao Claude/GPT**:

```
"DARWIN, me mostre um resumo completo do sistema"
```

**Claude consulta múltiplos endpoints**:

```
GET /api/v1/health              → Status geral
GET /api/v1/memory/profile      → Seu perfil
GET /api/v1/models/list         → Modelos disponíveis
GET /api/v1/corpus/training-status → Treinamento

Retorno consolidado:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 DARWIN 2.0 STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🟢 Sistema: Healthy
   - Core: operational
   - Pulsar: operational
   - gRPC: operational
   - Agentic: operational

👤 Seu Perfil:
   - Interações totais: 342
   - Expertise: biomaterials, research
   - Aprendendo: mathematics, quantum
   - Conexões: biomaterials ↔ mathematics (15x)

🤖 Modelos (11 ativos):
   - 6 DARWIN especializados
   - 4 modelos gerais
   - 1 embedding

🎓 Treinamento:
   - biomaterials: 98/100 (pronto em breve!)
   - mathematics: 45/100
   - Auto-training: ATIVO
```

---

## ⚡ Operação Zero-Terminal

**Você NUNCA precisa usar terminal** após deploy!

Tudo via conversação:
- ✅ Salvar memórias → conversa
- ✅ Buscar conhecimento → conversa
- ✅ Adicionar papers → conversa
- ✅ Treinar modelos → conversa
- ✅ Ver status → conversa
- ✅ Gerenciar modelos → conversa

**Terminal apenas para**:
- Deploy inicial (`./deploy.sh`)
- Debug (se algo quebrar)
- Monitoramento (Grafana)

---

## 🎉 Conclusão

DARWIN 2.0 é seu **segundo cérebro digital conversacional**:

✅ Opera via Claude/Custom GPT (natural)  
✅ Aprende com TODAS suas conversas  
✅ Ingere papers automaticamente  
✅ Treina modelos especializados  
✅ Evolui continuamente com você  
✅ Zero necessidade de terminal pós-deploy  

**Use naturalmente - DARWIN cuida de tudo!** 🧠✨


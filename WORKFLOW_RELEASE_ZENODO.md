# 🚀 WORKFLOW RELEASE + ZENODO - Darwin Core v2.0.0

**Status:** Zenodo configurado ✅  
**Ação:** Criar Release para gerar DOI automático

---

## 🎯 FLUXO COMPLETO (10 minutos)

### PASSO 1: Criar GitHub Release (3 min) ← VOCÊ AGORA!

**URL:** https://github.com/agourakis82/darwin-core/releases/new

**Configurações:**

```
Tag: v2.0.0 (select existing tag)

Release title:
Darwin Core v2.0.0 - Production-Ready AI Platform

Description:
[Copiar conteúdo completo de RELEASE_v2.0.0_DESCRIPTION.md]

☑️ Set as the latest release

[Publish release]
```

---

### PASSO 2: Webhook Zenodo (automático, ~30 seg)

**O que acontece:**
1. GitHub detecta novo Release
2. Webhook notifica Zenodo
3. Zenodo inicia snapshot do código

**Você verá no Zenodo:**
- "Processing..." (aguarde)

---

### PASSO 3: Zenodo Gera DOI (automático, 5-10 min)

**Zenodo cria:**
- ✅ Snapshot permanente do código (CERN)
- ✅ DOI único: `10.5281/zenodo.XXXXXXX`
- ✅ Página pública com metadados
- ✅ Citação automática

**Email:**
Você receberá email do Zenodo com o DOI!

---

### PASSO 4: Atualizar README com DOI (2 min) ← DEPOIS

**Após receber DOI do Zenodo:**

```bash
cd ~/workspace/darwin-core

# Editar README.md
# Adicionar badge no topo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

# Commit + push
git add README.md
git commit -m "docs: Add Zenodo DOI badge"
git push origin main
```

---

## 📊 TIMELINE ESPERADO

```
00:00 - Você cria Release v2.0.0 no GitHub ✓
00:01 - Webhook notifica Zenodo
00:02 - Zenodo inicia snapshot
00:05 - Zenodo processa código
00:10 - DOI gerado! Email enviado ✉️
00:12 - Você atualiza README com DOI badge
00:15 - COMPLETO! Darwin Core com DOI permanente 🎉
```

---

## ✅ CHECKLIST

**GitHub:**
- [x] Código pushed
- [x] Tag v2.0.0 pushed
- [x] Zenodo configurado
- [ ] Release v2.0.0 criado ← AGORA!

**Zenodo (automático):**
- [ ] Webhook recebido
- [ ] Snapshot criado
- [ ] DOI gerado
- [ ] Email recebido

**Finalização:**
- [ ] README atualizado com DOI
- [ ] Badge verificado

---

## 🎊 RESULTADO FINAL

**GitHub Release:**
https://github.com/agourakis82/darwin-core/releases/tag/v2.0.0

**Zenodo DOI:** (após 5-10 min)
https://doi.org/10.5281/zenodo.XXXXXXX

**Citação:**
```
Dr. Demetrios Agourakis. (2025). Darwin Core v2.0.0 - AI Platform. 
Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## 🚀 AÇÃO IMEDIATA

**AGORA:** Criar Release no GitHub!
1. Abrir: https://github.com/agourakis82/darwin-core/releases/new
2. Copiar conteúdo de: RELEASE_v2.0.0_DESCRIPTION.md
3. Publish!

**Zenodo fará o resto automaticamente! 🎉**

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**


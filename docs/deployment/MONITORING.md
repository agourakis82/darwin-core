# Monitoring Setup - Darwin Core

**Stack:** Prometheus + Grafana + Loki  
**Version:** 2.0.0  
**Status:** Production-ready

---

## 🎯 Overview

Darwin Core includes comprehensive monitoring with:
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Loki**: Log aggregation (optional)
- **OpenTelemetry**: Distributed tracing

---

## 🚀 Quick Setup

```bash
# Deploy monitoring stack
kubectl apply -f .darwin/cluster/k8s/prometheus.yaml
kubectl apply -f .darwin/cluster/k8s/grafana.yaml

# Verify
kubectl get pods -n darwin-core -l app=prometheus
kubectl get pods -n darwin-core -l app=grafana

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n darwin-core

# Open: http://localhost:3000
# User: admin
# Password: darwin2025 (change in production!)
```

---

## 📊 Prometheus Metrics

### Application Metrics

**Exposed at:** `http://darwin-core:9090/metrics`

**Key metrics:**

```prometheus
# Request metrics
darwin_requests_total{method, endpoint, status}
darwin_request_duration_seconds{method, endpoint}

# RAG++ metrics
darwin_rag_query_total{variant, status}
darwin_rag_query_duration_seconds{variant, quantile}
darwin_graphrag_entities_extracted_total
darwin_selfrag_reflection_decisions_total{decision}

# Multi-AI metrics
darwin_ai_routing_decisions_total{ai_provider, domain}
darwin_ai_chat_duration_seconds{ai_provider, quantile}
darwin_multi_ai_fallback_total{from_ai, to_ai}

# Embedding metrics
darwin_embedding_requests_total{model}
darwin_embedding_duration_seconds{model, quantile}
darwin_embedding_cache_hits_total{model}
darwin_embedding_cache_misses_total{model}

# Cache metrics
darwin_cache_hit_ratio{layer}  # L1, L2, semantic
darwin_cache_size_bytes{layer}
darwin_cache_evictions_total{layer}

# System metrics
darwin_active_connections
darwin_memory_usage_bytes
darwin_cpu_usage_percent
```

### Kubernetes Metrics

**Via kube-state-metrics:**

```prometheus
# Pod metrics
kube_pod_status_phase{namespace="darwin-core"}
kube_pod_container_status_restarts_total{namespace="darwin-core"}

# Resource usage
container_cpu_usage_seconds_total{namespace="darwin-core"}
container_memory_usage_bytes{namespace="darwin-core"}
```

---

## 📈 Grafana Dashboards

### Darwin Core Overview

**Panels:**
1. Request Rate (req/s)
2. Request Duration (p50, p95, p99)
3. Error Rate
4. Active Pods
5. CPU Usage (%)
6. Memory Usage (%)
7. Cache Hit Ratio
8. RAG Query Duration

**Queries:**
```promql
# Request rate
rate(darwin_requests_total[5m])

# p95 latency
histogram_quantile(0.95, darwin_request_duration_seconds_bucket)

# Error rate
rate(darwin_requests_total{status=~"5.."}[5m]) / rate(darwin_requests_total[5m])

# Cache hit ratio
darwin_cache_hit_ratio{layer="semantic"}
```

### RAG++ Performance

**Panels:**
1. GraphRAG Queries/min
2. Self-RAG Accuracy
3. Embedding Performance
4. Vector Search Duration
5. Cache Efficiency

### Multi-AI Orchestration

**Panels:**
1. Routing Decisions (by AI)
2. AI Response Times
3. Fallback Rate
4. Domain Distribution

### Resource Utilization

**Panels:**
1. CPU Usage (per pod)
2. Memory Usage (per pod)
3. Network I/O
4. Disk I/O
5. HPA Status

---

## 🔔 Alerts

### Prometheus Alert Rules

**Create:** `.darwin/cluster/k8s/prometheus-rules.yaml`

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-rules
  namespace: darwin-core
data:
  alerts.yml: |
    groups:
      - name: darwin-core
        interval: 30s
        rules:
          # High error rate
          - alert: HighErrorRate
            expr: rate(darwin_requests_total{status=~"5.."}[5m]) > 0.1
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High error rate detected"
              description: "Error rate is {{ $value }} req/s"
          
          # High latency
          - alert: HighLatency
            expr: histogram_quantile(0.95, darwin_request_duration_seconds_bucket) > 2.0
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High request latency"
              description: "p95 latency is {{ $value }}s"
          
          # Pod not ready
          - alert: PodNotReady
            expr: kube_pod_status_phase{namespace="darwin-core", phase!="Running"} == 1
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "Pod not ready"
              description: "Pod {{ $labels.pod }} is not running"
          
          # High memory usage
          - alert: HighMemoryUsage
            expr: container_memory_usage_bytes{namespace="darwin-core"} / container_spec_memory_limit_bytes > 0.9
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High memory usage"
              description: "Memory usage is {{ $value | humanizePercentage }}"
          
          # Cache low hit ratio
          - alert: LowCacheHitRatio
            expr: darwin_cache_hit_ratio < 0.5
            for: 10m
            labels:
              severity: info
            annotations:
              summary: "Low cache hit ratio"
              description: "Cache hit ratio is {{ $value | humanizePercentage }}"
```

---

## 📋 Logging

### Structured Logs (JSON)

**Format:**
```json
{
  "timestamp": "2025-11-05T19:30:00.123Z",
  "level": "INFO",
  "logger": "darwin.core",
  "message": "GraphRAG query completed",
  "duration_ms": 234,
  "query_type": "global",
  "entities_found": 42,
  "trace_id": "abc123...",
  "span_id": "def456..."
}
```

### Log Levels

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("darwin.core")

# Usage
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred", exc_info=True)
```

### Viewing Logs

**kubectl:**
```bash
# Follow logs
kubectl logs -f deployment/darwin-core -n darwin-core

# Last 100 lines
kubectl logs --tail=100 deployment/darwin-core -n darwin-core

# Logs from all pods
kubectl logs -l app=darwin-core -n darwin-core --all-containers=true

# Previous pod logs (if crashed)
kubectl logs deployment/darwin-core -n darwin-core --previous
```

**Grafana Loki (if installed):**
```
# Query syntax
{namespace="darwin-core", app="darwin-core"} |= "ERROR"
{namespace="darwin-core"} |= "GraphRAG"
{namespace="darwin-core"} | json | level="ERROR"
```

---

## 🔍 Tracing (OpenTelemetry)

### Setup

**Environment variables (in deployment.yaml):**
```yaml
env:
- name: OTEL_ENABLED
  value: "true"
- name: OTEL_ENDPOINT
  value: "otel-collector.monitoring.svc.cluster.local:4317"
- name: OTEL_SERVICE_NAME
  value: "darwin-core"
```

### Viewing Traces

**Jaeger UI:**
```bash
kubectl port-forward svc/jaeger-query 16686:16686 -n monitoring

# Open: http://localhost:16686
# Search for: darwin-core service
```

**Trace Example:**
```
Request → API Gateway → GraphRAG → Qdrant → LLM → Response
  1ms       5ms           150ms      50ms    2000ms    2206ms total
```

---

## 📊 Dashboards

### Import Dashboards

```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n darwin-core

# Login: admin / darwin2025

# Import dashboard:
# 1. Go to Dashboards → Import
# 2. Upload JSON from .darwin/cluster/k8s/grafana.yaml
# 3. Select Prometheus datasource
# 4. Import
```

### Available Dashboards

1. **Darwin Core Overview**
   - Request rate, latency, errors
   - Pod status, resource usage
   - Cache performance

2. **RAG++ Performance**
   - GraphRAG queries, entities
   - Self-RAG accuracy, reflections
   - Embedding performance
   - Vector search duration

3. **Multi-AI Orchestration**
   - Routing decisions by AI
   - AI response times
   - Fallback rate
   - Domain distribution

4. **Kubernetes Resources**
   - CPU usage per pod
   - Memory usage per pod
   - Network I/O
   - HPA status

---

## 🚨 Alerting

### Alert Channels

**Slack:**
```yaml
# In prometheus-config
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

# AlertManager config
receivers:
  - name: slack
    slack_configs:
      - api_url: YOUR_SLACK_WEBHOOK
        channel: '#darwin-alerts'
        title: 'Darwin Core Alert'
```

**Email:**
```yaml
receivers:
  - name: email
    email_configs:
      - to: 'agourakis@agourakis.med.br'
        from: 'alerts@darwin.ai'
        smarthost: 'smtp.gmail.com:587'
```

---

## 🔧 Troubleshooting

### Prometheus not scraping

```bash
# Check pod annotations
kubectl get pod -n darwin-core -o yaml | grep prometheus

# Should see:
# prometheus.io/scrape: "true"
# prometheus.io/port: "9090"
# prometheus.io/path: "/metrics"

# Check Prometheus targets
kubectl port-forward svc/prometheus 9090:9090 -n darwin-core
# Open: http://localhost:9090/targets
```

### Grafana no data

```bash
# Check datasource
# Grafana → Configuration → Data Sources → Prometheus
# Test: Should be "Data source is working"

# Check dashboards
# Verify query syntax
# Check time range (Last 1 hour)
```

### High resource usage

```bash
# Check Prometheus retention
kubectl edit deployment/prometheus -n darwin-core
# Adjust: --storage.tsdb.retention.time=15d

# Check Grafana queries
# Reduce scrape interval
# Optimize dashboard queries
```

---

## 📞 Support

**Documentation:**
- Architecture: [ARCHITECTURE.md](../architecture/ARCHITECTURE.md)
- Cluster Setup: [CLUSTER_SETUP.md](CLUSTER_SETUP.md)

**Issues:**
- GitHub: https://github.com/agourakis82/darwin-core/issues

---

**"Ciência rigorosa. Resultados honestos. Impacto real."**


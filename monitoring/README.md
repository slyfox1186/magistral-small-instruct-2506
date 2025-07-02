# 📊 Neural Consciousness Monitoring Stack

**Phase 3A: Baseline Instrumentation & Observability**

This directory contains the monitoring infrastructure for the Neural Consciousness Chat System, implementing Gemini's recommended "80/20 Observability" approach.

## 🎯 **Purpose: Establish Baseline Before Optimization**

Before implementing the Unified Inference Service (Phase 3B), we need quantitative baselines to:
- Measure current GPU utilization inefficiencies  
- Track response latency patterns
- Monitor metacognitive evaluation performance
- Validate improvements with real data

## 🚀 **Quick Start**

### 1. **Start Monitoring Stack**
```bash
cd monitoring
docker-compose up -d
```

### 2. **Verify Metrics Endpoint**
```bash
curl http://localhost:8000/metrics
```

### 3. **Access Dashboards**
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/neural-consciousness)

## 📈 **Golden Signals Tracked**

### **1. LATENCY**
- `neural_chat_request_duration_seconds` - P50, P95, P99 response times
- `neural_chat_metacognitive_duration_seconds` - Time spent on self-evaluation

### **2. TRAFFIC** 
- `neural_chat_requests_total` - Request rate by endpoint and status
- `neural_chat_metacognitive_evaluations_total` - Self-evaluation rate

### **3. ERRORS**
- `neural_chat_errors_total` - Error rate by type and endpoint

### **4. SATURATION**
- `neural_chat_gpu_queue_depth` - GPU processing queue by priority
- `neural_chat_model_lock_held` - GPU lock status (1=held, 0=free)

### **5. NEURAL CONSCIOUSNESS SPECIFIC**
- `neural_chat_response_quality_score` - Quality scores by dimension
- Quality dimensions: factual_accuracy, relevance, completeness, clarity, coherence, helpfulness, confidence

## 🎛️ **Grafana Dashboard**

The baseline dashboard shows:
- **Request Rate**: Real-time RPS 
- **Response Latency**: P95 latency with thresholds (green<2s, yellow<5s, red>5s)
- **GPU Queue Depth**: Priority queue visualization
- **Metacognitive Evaluation Rate**: Self-improvement frequency  
- **Response Quality Scores**: AI consciousness quality metrics
- **Error Rate**: System reliability metrics

## 🔍 **Key Metrics to Watch**

**CRITICAL for Phase 3 Planning:**

1. **GPU Queue Depth** - When this spikes >3, metacognitive evaluation is skipped
2. **P95 Latency** - Current baseline to beat with Unified Inference Service
3. **Metacognitive Rate vs Request Rate** - How often we achieve "consciousness"
4. **Quality Score Distribution** - Are we actually improving responses?

## 📊 **Expected Baseline Metrics**

Based on current architecture analysis:

- **P95 Latency**: 3-8 seconds (GPU contention dependent)
- **GPU Queue Depth**: Frequently >3 during load (triggers heuristic-only mode)
- **Metacognitive Rate**: <50% of requests get full LLM evaluation
- **Quality Improvement**: ~20-30% of evaluations result in response refinement

## 🎯 **Success Criteria for Phase 3B**

After implementing Unified Inference Service, we should see:

- **P95 Latency**: <2 seconds (5-20x throughput improvement)
- **GPU Queue Depth**: Consistently <2 (efficient batching)
- **Metacognitive Rate**: >90% of requests get full evaluation
- **Quality Improvement**: Maintained or improved despite higher throughput

## 🔧 **Troubleshooting**

### Metrics Not Appearing
```bash
# Check if prometheus_client is installed
python -c "import prometheus_client; print('OK')"

# Verify /metrics endpoint responds
curl -v http://localhost:8000/metrics

# Check Docker containers
docker-compose logs prometheus
docker-compose logs grafana
```

### Dashboard Not Loading
- Verify Grafana is accessible: http://localhost:3000
- Default credentials: admin / neural-consciousness
- Dashboard should auto-load, check provisioning logs

### No Data in Graphs
- Send test requests to generate metrics:
```bash
curl -X POST http://localhost:8000/api/chat-stream \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"session_id":"test"}'
```

## 📁 **Files**

- `prometheus.yml` - Prometheus configuration
- `docker-compose.yml` - Complete monitoring stack  
- `grafana-dashboard.json` - Baseline dashboard configuration
- `README.md` - This documentation

## 🚀 **Next Steps: Phase 3B**

Once baseline metrics are captured (run system for 24-48 hours), proceed to:
1. **Unified Inference Service** implementation
2. **A/B validation** using these same metrics
3. **Performance optimization** guided by dashboard data

This monitoring foundation will be essential for validating all future improvements to the Neural Consciousness system.
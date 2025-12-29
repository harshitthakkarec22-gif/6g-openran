# Management & Orchestration

This directory contains the management and orchestration layer for the 6G OPENRAN system.

## Overview

The management layer provides centralized control, monitoring, and lifecycle management for all network functions:

```
┌─────────────────────────────────────────────────────────┐
│         Service Management & Orchestration (SMO)        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Orchestrator │  │  Monitoring  │  │     Logs     │ │
│  │    (MANO)    │  │ (Prometheus) │  │     (ELK)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼──────────────────┼──────────────────┼─────────┘
          │                  │                  │
          ▼                  ▼                  ▼
    Network Functions    Telemetry          Log Data
```

## Components

### Orchestrator

Location: `management/orchestrator/`

**Purpose**: Lifecycle management and orchestration of network functions

**Key Functions**:
- Network function onboarding
- Resource allocation
- Scaling (horizontal and vertical)
- Fault management
- Configuration management

**Technologies**:
- Kubernetes for container orchestration
- Helm for package management
- Terraform for infrastructure as code

### Monitoring

Location: `management/monitoring/`

**Purpose**: Real-time monitoring and metrics collection

**Components**:
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification

**Metrics Collected**:
- System metrics (CPU, memory, network)
- Application metrics (throughput, latency)
- Business metrics (active users, sessions)

### Logging

Location: `management/logs/`

**Purpose**: Centralized logging and log analysis

**Stack (ELK)**:
- **Elasticsearch**: Log storage and indexing
- **Logstash**: Log processing and enrichment
- **Kibana**: Log visualization and search

## Setup

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installations
kubectl version --client
helm version
```

### Deploy Monitoring Stack

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace

# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials: admin/prom-operator
```

### Deploy Logging Stack

```bash
# Add Elastic Helm repo
helm repo add elastic https://helm.elastic.co
helm repo update

# Install Elasticsearch
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --create-namespace

# Install Kibana
helm install kibana elastic/kibana \
  --namespace logging

# Install Filebeat for log collection
helm install filebeat elastic/filebeat \
  --namespace logging

# Access Kibana
kubectl port-forward -n logging svc/kibana-kibana 5601:5601
```

## Usage

### Deploying Network Functions

```bash
# Deploy using Helm
helm install ran-cu ./charts/ran-cu \
  --values configs/ran/cu-values.yaml

# Check deployment status
kubectl get pods -l app=ran-cu

# View logs
kubectl logs -f deployment/ran-cu
```

### Monitoring Dashboard

Access Grafana at `http://localhost:3000`

**Pre-configured Dashboards**:
1. **System Overview**: CPU, memory, network usage
2. **RAN Metrics**: Cell load, throughput, connected UEs
3. **Core Network**: Active sessions, signaling load
4. **RIC**: xApp performance, control loop latency

### Viewing Logs

Access Kibana at `http://localhost:5601`

**Common Queries**:
```
# All errors in last hour
level:ERROR AND @timestamp:[now-1h TO now]

# Logs from specific component
component:AMF

# Failed registration attempts
message:"Registration failed"
```

## Configuration

### Prometheus Configuration

```yaml
# management/monitoring/prometheus-config.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ran-cu'
    static_configs:
      - targets: ['ran-cu:9090']
  
  - job_name: 'core-amf'
    static_configs:
      - targets: ['core-amf:9091']
  
  - job_name: 'ric'
    static_configs:
      - targets: ['ric:9092']
```

### Grafana Dashboards

Import dashboards from `management/monitoring/dashboards/`

### Alert Rules

```yaml
# management/monitoring/alerts.yaml
groups:
  - name: core_network
    interval: 30s
    rules:
      - alert: HighCPUUsage
        expr: cpu_usage_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
      
      - alert: SessionEstablishmentFailed
        expr: rate(session_failures[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High session failure rate"
```

## API Reference

### Orchestrator API

```python
# Example: Deploy network function
import requests

def deploy_nf(nf_type, config):
    response = requests.post(
        "http://orchestrator:8080/api/v1/nf/deploy",
        json={
            "type": nf_type,
            "config": config
        }
    )
    return response.json()

# Deploy CU
result = deploy_nf("ran-cu", {
    "replicas": 1,
    "resources": {
        "cpu": "2",
        "memory": "4Gi"
    }
})
```

### Monitoring API

```python
# Query Prometheus
def get_metric(query):
    response = requests.get(
        "http://prometheus:9090/api/v1/query",
        params={"query": query}
    )
    return response.json()

# Get current CPU usage
cpu_usage = get_metric("cpu_usage_percent")
```

## Troubleshooting

### Prometheus Not Scraping

```bash
# Check Prometheus targets
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090
# Visit http://localhost:9090/targets

# Verify network policies
kubectl get networkpolicies -A
```

### High Memory Usage

```bash
# Check resource usage
kubectl top pods -A

# Increase resource limits
kubectl edit deployment <deployment-name>

# Add resource limits
resources:
  limits:
    cpu: "4"
    memory: "8Gi"
  requests:
    cpu: "2"
    memory: "4Gi"
```

### Logs Not Appearing

```bash
# Check Filebeat status
kubectl logs -n logging daemonset/filebeat

# Verify Elasticsearch
kubectl port-forward -n logging svc/elasticsearch-master 9200:9200
curl http://localhost:9200/_cluster/health
```

## Best Practices

1. **Resource Management**
   - Set appropriate resource requests and limits
   - Use resource quotas per namespace
   - Monitor resource utilization

2. **High Availability**
   - Deploy multiple replicas for critical components
   - Use pod disruption budgets
   - Distribute pods across nodes

3. **Monitoring**
   - Define comprehensive metrics
   - Set up alerting rules
   - Regular dashboard reviews

4. **Logging**
   - Use structured logging (JSON)
   - Include context (request ID, user ID)
   - Set appropriate log levels

5. **Security**
   - Use RBAC for access control
   - Encrypt sensitive data
   - Regular security audits

## References

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Elastic Stack Documentation](https://www.elastic.co/guide/)

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

## License

See [LICENSE](../../LICENSE) file in the root directory.

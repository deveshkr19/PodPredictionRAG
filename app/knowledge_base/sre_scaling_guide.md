## Kubernetes/OpenShift Pod Scaling Guidelines

- Always set resource `requests` and `limits`.
- Use Horizontal Pod Autoscaler (HPA) based on CPU/Memory.
- Monitor pod eviction events.
- Use `readinessProbe` and `livenessProbe`.

## Load Testing Metrics to Monitor
- TPS (transactions per second)
- CPU & Memory utilization
- Error % and Response Time
- GC and heap trends

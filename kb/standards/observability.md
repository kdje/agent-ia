# Observability Standards (SAP S/4 RISE + MuleSoft)

## 1) Logging (structuré)
- Logs structurés (JSON si possible).
- Champs minimum :
  - timestamp, level, service, environment
  - correlationId
  - operation / interfaceName
  - status (SUCCESS/FAIL)
  - durationMs
- Ne pas logger de PII.

## 2) Metrics
- KPIs minimum par interface :
  - throughput (req/min, msg/min)
  - latency (p50/p95/p99)
  - error rate (4xx/5xx ou TECH/FUNC)
  - retries count
  - DLQ count
- Suivre les quotas/rate limits si exposé.

## 3) Tracing (E2E)
- Propager trace/correlation sur chaque hop (API, broker, backend).
- Permettre un diagnostic “du front jusqu’à SAP” avec un seul ID.

## 4) Alerting
- Alertes sur :
  - error rate anormal
  - DLQ > seuil
  - latence > SLO
  - saturation (threads, CPU, memory)
- Différencier alertes “action immédiate” vs “à traiter”.

## 5) Runbooks
- Pour chaque interface critique : runbook de triage (symptômes, checks, actions).
- Inclure le chemin pour retrouver logs/metrics/traces avec correlationId.

# Integration Standards (SAP S/4 RISE + MuleSoft)

## 1) Correlation / Trace ID (E2E)
- Chaque flux DOIT porter un identifiant de corrélation unique de bout en bout.
- Propager le `correlationId` dans :
  - headers HTTP (ex: `X-Correlation-Id`)
  - message attributes (event/broker)
  - logs et métriques
- Si absent à l’entrée, le générer et le renvoyer dans la réponse.

## 2) Idempotence
- Tout endpoint/consumer DOIT être idempotent quand le retry/replay est possible.
- Stratégie recommandée :
  - clé d’idempotence = identifiant métier stable (ex: `orderId + version` / `documentNo + timestamp` selon le cas)
  - stockage d’état (cache/DB) avec TTL aligné sur la fenêtre de retry (ex: 24–72h)
  - si même clé reçue → retourner le même résultat / ignorer le doublon (selon contrat)

## 3) Retry / Backoff
- Retry uniquement sur erreurs techniques/transitoires (timeout, 5xx, réseau).
- Backoff exponentiel + jitter recommandé.
- Fixer un max attempts + un max duration.
- Ne PAS retry les erreurs fonctionnelles (4xx métier, validation).

## 4) Gestion des erreurs (technique vs fonctionnelle)
- Séparer explicitement :
  - erreurs fonctionnelles (validation, règles métier, données incohérentes)
  - erreurs techniques (infra, auth, réseau, timeout)
- Contrat d’erreur standard :
  - `errorCode`
  - `message`
  - `category` (FUNCTIONAL/TECHNICAL)
  - `correlationId`
  - `details` (liste)
- Les erreurs fonctionnelles doivent être traçables et actionnables par le support métier.

## 5) DLQ / Reprocess / Replay
- Pour l’asynchrone, prévoir une DLQ (dead-letter) ou un mécanisme équivalent.
- Le replay doit être :
  - contrôlé (par lot / par clé)
  - audité (qui, quand, pourquoi)
  - idempotent (cf. section 2)
- Conserver la charge utile minimale nécessaire au diagnostic (attention PII).

## 6) Timeouts / Limits
- Fixer des timeouts explicites (connect/read/overall) selon SLA.
- Définir des limites :
  - taille max payload
  - pagination
  - rate limiting si exposition externe

## 7) Versioning des contrats
- APIs : versioning explicite (URI ou header) + règles de compatibilité.
- Events : schema versionné (ex: `schemaVersion`) + stratégie de cohabitation.

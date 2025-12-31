# Security Standards (SAP S/4 RISE + MuleSoft)

## 1) AuthN/AuthZ
- APIs : OAuth2 recommandé (client credentials pour system-to-system).
- Éviter les secrets statiques en dur ; utiliser un vault/secret manager.
- Scopes / permissions minimales (least privilege).

## 2) Transport Security
- TLS partout (HTTPS).
- mTLS si exigence forte (externe, partenaire, flux sensibles).
- Rotation régulière des certificats + monitoring d’expiration.

## 3) Secrets Management
- Aucun secret dans le code, les logs, ou les configs en clair.
- Stocker dans un gestionnaire de secrets + injection runtime.

## 4) Données sensibles / PII
- Minimiser les PII dans les payloads si possible.
- Masquer/anonymiser les champs sensibles dans les logs.
- Chiffrement au repos si stockage (DLQ, persistance idempotence).

## 5) Audit
- Journaliser : accès, changements de config, replay, opérations critiques.
- Toujours inclure `correlationId` dans les événements d’audit.

## 6) Sécurité MuleSoft (si applicable)
- Appliquer les policies API Manager :
  - rate limiting / spike control
  - client-id enforcement / OAuth2
- Séparer environnements et droits (dev/test/prod).

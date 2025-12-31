# Utiliser `cli.py` (Integration Architect Copilot)

Ce guide explique comment lancer la CLI pour générer un **JSON d’architecture** (options + design doc + checklist + plan de tests + ADR) en choisissant le modèle (**Qwen 14B** ou **Mistral Nemo 12B**).

> Emplacement attendu : `integration-copilot/app/cli.py`  
> Les commandes ci-dessous sont données pour **PowerShell** (Windows).

---

## 1) Prérequis (à vérifier une seule fois)

### 1.1 Qdrant tourne
```powershell
docker ps
```
Tu dois voir un conteneur `qdrant/qdrant`.

### 1.2 Les modèles Ollama sont présents
```powershell
ollama list
```
Doit inclure au minimum :
- `qwen2.5:14b-instruct`
- `mistral-nemo:12b-instruct`
- `bge-m3` (embeddings)

### 1.3 Ta KB est ingérée
Depuis `integration-copilot/app` :
```powershell
py ingest.py
```

---

## 2) Exécuter la commande principale `design`

### 2.1 Exemple complet avec Qwen 14B
Depuis `integration-copilot/app` :
```powershell
py cli.py design `
  --model "qwen2.5:14b-instruct" `
  --context "Synchroniser les commandes entre S/4HANA RISE et un WMS. Near-real-time, audit, idempotence." `
  --systems "SAP S/4 RISE, MuleSoft, WMS" `
  --constraints "SLA 99.9%;50k commandes/jour;PII;replay possible" `
  --data-objects "Sales Order, Delivery, Status" `
  --triggers "event"
```

### 2.2 Exemple rapide avec Mistral Nemo 12B
```powershell
py cli.py design `
  --model "mistral-nemo:12b-instruct" `
  --context "Même besoin" `
  --systems "SAP S/4 RISE, MuleSoft, WMS"
```

---

## 3) Formats attendus pour les arguments

- `--systems` : séparés par **virgules**
  - Exemple : `"SAP S/4 RISE, MuleSoft, WMS"`
- `--data-objects` : séparés par **virgules**
  - Exemple : `"Sales Order, Delivery"`
- `--constraints` : séparées par **point-virgule** `;`
  - Exemple : `"SLA 99.9%;latence < 2s;PII"`
- `--triggers` : séparés par **virgules**
  - Exemple : `"event, api"`

---

## 4) Définir un modèle par défaut (pour ne plus taper `--model`)

Dans PowerShell (session courante) :
```powershell
$env:COPILOT_MODEL="qwen2.5:14b-instruct"
```

Ensuite :
```powershell
py cli.py design --context "..." --systems "..."
```

---

## 5) Sauvegarder la sortie JSON dans un fichier

```powershell
py cli.py design --model "qwen2.5:14b-instruct" --context "..." --systems "..." |
  Out-File -Encoding utf8 result.json
```

---

## 6) Changer l’URL Qdrant (si besoin)

Ton `copilot.py` lit `QDRANT_URL`. Exemple :
```powershell
$env:QDRANT_URL="http://localhost:6333"
py cli.py design --context "..." --systems "..."
```

---

## 7) Aide intégrée

```powershell
py cli.py --help
py cli.py design --help
```

---

## 8) Checklist de dépannage rapide

- **`retrieve_context` retourne `[]`**  
  → ta KB est vide ou pas ingérée : relancer `py ingest.py` et vérifier que `kb/` contient des `.md/.txt`.

- **Erreur Qdrant (connexion/HTTP)**  
  → vérifier Qdrant : `docker ps` et que `http://localhost:6333` est accessible.

- **Modèle introuvable**  
  → `ollama list` puis `ollama pull <model>`.


nvidia-smi
Wed Dec 31 15:06:22 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.83                 Driver Version: 581.83         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 3000 Blac...  WDDM  |   00000000:01:00.0 Off |                  Off |
| N/A   37C    P8              8W /   55W |      84MiB /  12227MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           35908    C+G   ...stman\app-11.77.2\Postman.exe      N/A      |
+-----------------------------------------------------------------------------------------+
(venv_312) PS C:\dev\ia>

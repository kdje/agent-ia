# Integration Architect Copilot — Setup open-source avec switch de modèle (Qwen 14B / Mistral Nemo 12B)

Objectif : un agent **local**, **GPU**, **open-source**, avec **RAG** (Qdrant + embeddings) et la possibilité de **switcher** facilement entre :
- `qwen2.5:14b-instruct`
- `mistral-nemo:12b-instruct`

---

## 1) Installer le runner + récupérer les 2 modèles

On utilise **Ollama** comme runtime local (GPU).

```bash
ollama pull qwen2.5:14b-instruct
ollama pull mistral-nemo:12b-instruct
```

Vérifier qu’ils sont bien installés :

```bash
ollama list
```

Test rapide (même prompt sur les 2) :

```bash
ollama run qwen2.5:14b-instruct "Propose 2 options d’architecture d’intégration S/4->WMS."
ollama run mistral-nemo:12b-instruct "Propose 2 options d’architecture d’intégration S/4->WMS."
```

Vérifier que le GPU travaille :

```bash
nvidia-smi -l 1
```

---

## 2) Lancer Qdrant (vector DB) en local

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## 3) Créer le projet + dépendances Python

Arbo conseillée :

```
integration-copilot/
  kb/
    standards/
    templates/
    examples/
  app/
    ingest.py
    copilot.py
    cli.py
```

Dépendances :

```bash
pip install qdrant-client sentence-transformers pydantic typer rich ollama
```

---

## 4) Préparer ta base de connaissance (RAG)

Mets des `.md` ou `.txt` dans `kb/` :
- `kb/standards/` : retries, idempotence, erreurs, corr-id, sécurité, observabilité
- `kb/templates/` : template design doc + ADR
- `kb/examples/` : exemples anonymisés de flux / payload / contrats

---

## 5) Ingestion des docs dans Qdrant (embeddings GPU)

Crée `app/ingest.py` :

```python
import uuid
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

COLLECTION = "integration_kb"

def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += (chunk_size - overlap)
    return [c.strip() for c in chunks if c.strip()]

def main(kb_dir: str = "../kb"):
    client = QdrantClient(url="http://localhost:6333")
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")  # embeddings GPU

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qmodels.VectorParams(size=1024, distance=qmodels.Distance.COSINE),
        )

    kb_path = Path(kb_dir)
    points = []

    for path in kb_path.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() not in [".txt", ".md"]:
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()

        for chunk, vec in zip(chunks, vectors):
            points.append(qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"source": str(path.relative_to(kb_path)), "text": chunk},
            ))

    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"✅ Ingested {len(points)} chunks into '{COLLECTION}'")
    else:
        print("⚠️ Aucun .txt/.md trouvé dans kb/")

if __name__ == "__main__":
    main()
```

Lance :

```bash
cd app
python ingest.py
```

---

## 6) Le Copilot (RAG + génération JSON) avec switch de modèle

On ajoute :
- un paramètre `model=...`
- des presets VRAM-safe (important avec ~12GB VRAM)

Crée `app/copilot.py` :

```python
import os, json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import ollama

COLLECTION = "integration_kb"

# ---- Presets VRAM-safe (~12GB) ----
MODEL_PRESETS = {
    "qwen2.5:14b-instruct": {"num_ctx": 4096, "temperature": 0.2},
    "mistral-nemo:12b-instruct": {"num_ctx": 6144, "temperature": 0.2},
}
DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "qwen2.5:14b-instruct")

class RequirementInput(BaseModel):
    context: str
    systems: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    data_objects: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)

class ArchitectureOption(BaseModel):
    name: str
    summary: str
    flow_steps: List[str]
    pros: List[str]
    cons: List[str]
    risks: List[str]
    when_to_choose: str

class OutputDoc(BaseModel):
    assumptions: List[str]
    open_questions: List[str]
    recommended_option: str
    options: List[ArchitectureOption]
    design_doc: Dict[str, Any]
    review_checklist: List[str]
    test_plan: List[str]
    adr: Dict[str, Any]
    references: List[str]

def retrieve_context(query: str, top_k: int = 8) -> List[Dict[str, str]]:
    client = QdrantClient(url="http://localhost:6333")
    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    hits = client.search(collection_name=COLLECTION, query_vector=qvec, limit=top_k)
    return [{"source": h.payload.get("source","unknown"), "text": h.payload.get("text","")} for h in hits]

SYSTEM_PROMPT = """Tu es Integration Architect Copilot (SAP S/4 RISE + MuleSoft).
But: produire des livrables d’architecture actionnables.
Réponds UNIQUEMENT en JSON valide conforme au schéma.
Si info manquante: hypothèses explicites + questions ouvertes (max 5).
Aucun texte hors JSON.
"""

def build_prompt(req: RequirementInput, kb: List[Dict[str,str]]) -> str:
    kb_block = "\n\n".join([f"- source: {k['source']}\n  excerpt: {k['text']}" for k in kb])
    schema = OutputDoc.model_json_schema()
    return f"""
INPUT:
{req.model_dump_json(indent=2, ensure_ascii=False)}

RAG:
{kb_block}

SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

Règles:
- 2 à 3 options d’architecture distinctes
- recommended_option = nom exact d’une option
- design_doc inclut: overview, interfaces, data_contracts, nfr, security, observability, error_handling, cutover
- references = liste des sources RAG réellement utilisées
"""

def call_llm(model: str, prompt: str) -> str:
    preset = MODEL_PRESETS.get(model, {"num_ctx": 4096, "temperature": 0.2})
    resp = ollama.chat(
        model=model,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":prompt}],
        options={"temperature": preset["temperature"], "num_ctx": preset["num_ctx"]},
    )
    return resp["message"]["content"]

def generate(req: RequirementInput, model: str = DEFAULT_MODEL) -> OutputDoc:
    query = f"{req.context}\nSystems: {', '.join(req.systems)}\nData: {', '.join(req.data_objects)}"
    kb = retrieve_context(query, top_k=8)
    prompt = build_prompt(req, kb)

    content = call_llm(model, prompt)

    # 1 retry si JSON invalide
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        fix = call_llm(model, "Corrige ce JSON pour qu'il soit valide et conforme au schéma. Renvoie uniquement le JSON.\n\n" + content)
        data = json.loads(fix)

    return OutputDoc.model_validate(data)
```

---

## 7) CLI : switcher de modèle en une option

Crée `app/cli.py` :

```python
import typer
from copilot import RequirementInput, generate, DEFAULT_MODEL

app = typer.Typer()

@app.command()
def design(
    context: str,
    systems: str = "",
    constraints: str = "",
    data_objects: str = "",
    triggers: str = "",
    model: str = DEFAULT_MODEL,
):
    req = RequirementInput(
        context=context,
        systems=[s.strip() for s in systems.split(",") if s.strip()],
        constraints=[s.strip() for s in constraints.split(";") if s.strip()],
        data_objects=[s.strip() for s in data_objects.split(",") if s.strip()],
        triggers=[s.strip() for s in triggers.split(",") if s.strip()],
    )
    out = generate(req, model=model)
    print(out.model_dump_json(indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app()
```

### Utilisation

**Qwen 14B** :

```bash
python cli.py design   --model "qwen2.5:14b-instruct"   --context "Sync commandes S/4 -> WMS, near-real-time, audit, idempotence."   --systems "SAP S/4 RISE, MuleSoft, WMS"   --constraints "SLA 99.9%;50k/j;PII"   --data-objects "Sales Order, Delivery"   --triggers "event"
```

**Mistral Nemo 12B** :

```bash
python cli.py design   --model "mistral-nemo:12b-instruct"   --context "Même besoin"   --systems "SAP S/4 RISE, MuleSoft, WMS"
```

---

## 8) Switch via variable d’environnement (pratique)

Définir le modèle par défaut :

```bash
set COPILOT_MODEL=qwen2.5:14b-instruct   # Windows (cmd)
# ou
export COPILOT_MODEL=qwen2.5:14b-instruct # Linux/macOS
```

Ensuite tu peux omettre `--model`.

---

## 9) Réglages VRAM-safe (recommandés pour ~12GB)

- **Qwen 14B** : `num_ctx=4096`
- **Mistral Nemo 12B** : `num_ctx=6144`
- `top_k` RAG : 6–10 max

Si OOM :
- baisse `num_ctx` (ex: 3072 / 4096)
- baisse `top_k` (ex: 6)
- réduis `chunk_size` (ex: 900)

---

## 10) Workflow conseillé au quotidien

- Itérations rapides / rédaction structurée : **Mistral Nemo 12B**
- Trade-offs lourds / décisions d’archi : **Qwen 14B**

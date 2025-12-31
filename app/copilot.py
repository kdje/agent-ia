import os, json, math, re
import urllib.request
import urllib.error
from typing import List, Dict, Any

from pydantic import BaseModel, Field
import ollama

# ----------------- Config -----------------
COLLECTION = "integration_kb"

# Embeddings via Ollama (local)
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")  # ollama pull bge-m3

# Qdrant REST URL
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# ---- Presets VRAM-safe (~12GB) ----
MODEL_PRESETS = {
    "qwen2.5:14b-instruct": {"num_ctx": 4096, "temperature": 0.2},
    "mistral-nemo:12b-instruct": {"num_ctx": 6144, "temperature": 0.2},
}
DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "qwen2.5:14b-instruct")

# ----------------- Schemas -----------------
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

# ----------------- Helpers JSON robustes -----------------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s

def _extract_json_object(s: str) -> str:
    """
    Extrait le premier objet JSON { ... } du texte.
    Utile si le modèle ajoute du texte autour ou des code fences.
    """
    s = _strip_code_fences(s)

    # Cas où le modèle renvoie plusieurs objets: on prend le premier bloc {...}
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]

    return s

def json_loads_lenient(s: str) -> dict:
    return json.loads(_extract_json_object(s))

# ----------------- Embeddings (Ollama local) -----------------
def l2_normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return vec
    return [x / n for x in vec]

def embed_query(text: str, model: str = EMBED_MODEL, normalize: bool = True) -> List[float]:
    """
    Embedding d'une requête via Ollama (local).
    """
    resp = ollama.embed(model=model, input=text)
    vec = resp["embeddings"][0]
    return l2_normalize(vec) if normalize else vec

# ----------------- Qdrant search via REST (compat 100%) -----------------
def _qdrant_search_http(collection_name: str, qvec: List[float], top_k: int) -> List[Dict[str, Any]]:
    """
    Recherche vectorielle via API REST Qdrant:
    POST /collections/{collection}/points/search
    """
    url = f"{QDRANT_URL}/collections/{collection_name}/points/search"
    body = {
        "vector": qvec,
        "limit": top_k,
        "with_payload": True,
    }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"Erreur HTTP Qdrant ({e.code}) sur {url}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Impossible d'appeler Qdrant sur {QDRANT_URL}. "
            f"Vérifie que Qdrant tourne (docker) et que le port 6333 est accessible. Détail: {e}"
        )

    return payload.get("result", []) or []

def retrieve_context(query: str, top_k: int = 8) -> List[Dict[str, str]]:
    """
    Recherche dans Qdrant avec embeddings bge-m3 via Ollama (cohérent avec ingest.py).
    """
    qvec = embed_query(query)
    hits = _qdrant_search_http(COLLECTION, qvec, top_k)

    out: List[Dict[str, str]] = []
    for h in hits:
        pl = h.get("payload", {}) or {}
        out.append({
            "source": pl.get("source", "unknown"),
            "text": pl.get("text", ""),
        })
    return out

# ----------------- Prompting / génération -----------------
SYSTEM_PROMPT = """Tu es Integration Architect Copilot (SAP S/4 RISE + MuleSoft).
But: produire des livrables d’architecture actionnables.
Réponds UNIQUEMENT en JSON valide conforme au schéma.
Si info manquante: hypothèses explicites + questions ouvertes (max 5).
Aucun texte hors JSON.
"""

def build_prompt(req: RequirementInput, kb: List[Dict[str, str]]) -> str:
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

    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": preset["temperature"],
            "num_ctx": preset["num_ctx"],
        },
    )

    # Force JSON si supporté par la version de la lib Ollama
    try:
        resp = ollama.chat(**kwargs, format="json")
    except TypeError:
        resp = ollama.chat(**kwargs)

    return resp["message"]["content"]

def generate(req: RequirementInput, model: str = DEFAULT_MODEL) -> OutputDoc:
    query = f"{req.context}\nSystems: {', '.join(req.systems)}\nData: {', '.join(req.data_objects)}"
    kb = retrieve_context(query, top_k=8)
    prompt = build_prompt(req, kb)

    content = call_llm(model, prompt)

    # Parse tolérant + 2 tentatives de réparation
    try:
        data = json_loads_lenient(content)
    except Exception:
        fix_prompt = (
            "Réécris la réponse en JSON STRICT et VALIDE, conforme au schéma. "
            "Aucun texte hors JSON, pas de ```.\n\n"
            "TEXTE À RÉPARER:\n"
            f"{content}"
        )
        fixed = call_llm(model, fix_prompt)

        try:
            data = json_loads_lenient(fixed)
        except Exception:
            fix_prompt2 = (
                "Tu dois produire uniquement un objet JSON valide. "
                "Ne mets rien d'autre. Corrige toutes les virgules, guillemets, etc.\n\n"
                "TEXTE À RÉPARER:\n"
                f"{fixed}"
            )
            fixed2 = call_llm(model, fix_prompt2)
            data = json_loads_lenient(fixed2)

    return OutputDoc.model_validate(data)

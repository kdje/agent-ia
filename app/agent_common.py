import os, json, math, re
import urllib.request, urllib.error
from typing import List, Dict, Any

import ollama

# ----------------- Config partagée -----------------
COLLECTION = os.getenv("QDRANT_COLLECTION", "integration_kb")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Embeddings via Ollama (local)
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# LLM presets (VRAM-safe)
MODEL_PRESETS = {
    "qwen2.5:14b-instruct": {"num_ctx": 4096, "temperature": 0.2},
    "mistral-nemo:latest": {"num_ctx": 6144, "temperature": 0.2},
    "mistral:latest": {"num_ctx": 4096, "temperature": 0.2},
}
DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "qwen2.5:14b-instruct")

# ----------------- JSON helpers -----------------
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
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s

def json_loads_lenient(s: str) -> dict:
    return json.loads(_extract_json_object(s))

# ----------------- Embeddings helpers -----------------
def l2_normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0:
        return vec
    return [x / n for x in vec]

def _sanitize_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return s.strip()

def embed_query(text: str, model: str = EMBED_MODEL, normalize: bool = True) -> List[float]:
    """
    Embedding d'une requête via Ollama (local). (1 texte -> faible risque NaN)
    """
    text = _sanitize_text(text)
    resp = ollama.embed(model=model, input=text)
    vec = resp["embeddings"][0]
    return l2_normalize(vec) if normalize else vec

# ----------------- Qdrant search via REST -----------------
def _qdrant_search_http(collection_name: str, qvec: List[float], top_k: int) -> List[Dict[str, Any]]:
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
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"Erreur HTTP Qdrant ({e.code}) sur {url}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Impossible d'appeler Qdrant sur {QDRANT_URL}. "
            f"Vérifie Qdrant (docker) + port 6333. Détail: {e}"
        )

    return payload.get("result", []) or []

def retrieve_context(query: str, top_k: int = 8) -> List[Dict[str, str]]:
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

def kb_block(kb: List[Dict[str, str]]) -> str:
    return "\n\n".join([f"- source: {k['source']}\n  excerpt: {k['text']}" for k in kb])

# ----------------- LLM call + repair -----------------
def call_llm(model: str, system_prompt: str, user_prompt: str) -> str:
    preset = MODEL_PRESETS.get(model, {"num_ctx": 4096, "temperature": 0.2})
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": preset["temperature"],
            "num_ctx": preset["num_ctx"],
        },
    )

    try:
        resp = ollama.chat(**kwargs, format="json")
    except TypeError:
        resp = ollama.chat(**kwargs)

    return resp["message"]["content"]

def parse_json_with_repairs(model: str, system_prompt: str, content: str, schema_json: dict) -> dict:
    """
    Parse lenient + 2 repairs si besoin.
    """
    try:
        return json_loads_lenient(content)
    except Exception:
        fix_prompt = (
            "Réécris la réponse en JSON STRICT et VALIDE, conforme au schéma. "
            "Aucun texte hors JSON, pas de ```.\n\n"
            "SCHEMA:\n"
            f"{json.dumps(schema_json, ensure_ascii=False)}\n\n"
            "TEXTE À RÉPARER:\n"
            f"{content}"
        )
        fixed = call_llm(model, system_prompt, fix_prompt)

        try:
            return json_loads_lenient(fixed)
        except Exception:
            fix_prompt2 = (
                "Tu dois produire uniquement un objet JSON valide. "
                "Ne mets rien d'autre. Corrige toutes les virgules, guillemets, etc.\n\n"
                "SCHEMA:\n"
                f"{json.dumps(schema_json, ensure_ascii=False)}\n\n"
                "TEXTE À RÉPARER:\n"
                f"{fixed}"
            )
            fixed2 = call_llm(model, system_prompt, fix_prompt2)
            return json_loads_lenient(fixed2)

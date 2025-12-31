import os
import contextlib
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

from mcp.server.fastmcp import FastMCP

# ---------------- Config ----------------
MCP_MOUNT_PATH = os.getenv("MCP_MOUNT_PATH", "/mcp")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

ENABLE_ATLASSIAN_TOOLS = os.getenv("ENABLE_ATLASSIAN_TOOLS", "false").lower() == "true"
ATLASSIAN_BASE_URL = os.getenv("ATLASSIAN_BASE_URL", "").rstrip("/")
ATLASSIAN_EMAIL = os.getenv("ATLASSIAN_EMAIL", "")
ATLASSIAN_API_TOKEN = os.getenv("ATLASSIAN_API_TOKEN", "")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY", "")
CONFLUENCE_PARENT_ID = os.getenv("CONFLUENCE_PARENT_ID", "")
JIRA_EPIC_NAME_FIELD = os.getenv("JIRA_EPIC_NAME_FIELD", "")  # ex: customfield_10011

qdrant = QdrantClient(url=QDRANT_URL)

# MCP server
mcp = FastMCP("agent-ia-mcp", stateless_http=True, json_response=True)

# ---------------- Helpers ----------------
def _atlassian_auth() -> httpx.BasicAuth:
    return httpx.BasicAuth(ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)

def _ensure_atlassian_enabled() -> None:
    if not ENABLE_ATLASSIAN_TOOLS:
        raise RuntimeError("Atlassian tools are disabled. Set ENABLE_ATLASSIAN_TOOLS=true")
    if not (ATLASSIAN_BASE_URL and ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN):
        raise RuntimeError("Missing ATLASSIAN_BASE_URL / ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN")

async def _ollama_post_raw(path: str, payload: Dict[str, Any], timeout_s: float = 120) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        return await client.post(f"{OLLAMA_URL}{path}", json=payload)

# ---------------- Tools ----------------
@mcp.tool()
async def ollama_generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Génération texte.
    - tente /api/generate (API Ollama "native")
    - si 404, fallback /v1/chat/completions (API OpenAI compatible)
    """
    payload_native: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload_native["system"] = system
    if options:
        payload_native["options"] = options

    r = await _ollama_post_raw("/api/generate", payload_native, timeout_s=180)

    if r.status_code == 404:
        # Fallback OpenAI-compatible
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        r2 = await _ollama_post_raw(
            "/v1/chat/completions",
            {"model": model, "messages": messages},
            timeout_s=180,
        )
        r2.raise_for_status()
        data2 = r2.json()
        text = data2.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"model": model, "response": text, "raw": data2}

    r.raise_for_status()
    data = r.json()
    return {"model": model, "response": data.get("response", ""), "raw": data}

@mcp.tool()
async def ollama_embed(model: str, input: str) -> Dict[str, Any]:
    """
    Embeddings.
    - tente /api/embeddings (API Ollama "native")
    - si 404, fallback /v1/embeddings (API OpenAI compatible)
    """
    r = await _ollama_post_raw("/api/embeddings", {"model": model, "prompt": input}, timeout_s=60)

    if r.status_code == 404:
        r2 = await _ollama_post_raw("/v1/embeddings", {"model": model, "input": input}, timeout_s=60)
        r2.raise_for_status()
        data2 = r2.json()
        emb = (data2.get("data") or [{}])[0].get("embedding", [])
        return {"model": model, "embedding": emb, "raw": data2}

    r.raise_for_status()
    data = r.json()
    return {"model": model, "embedding": data.get("embedding", []), "raw": data}

@mcp.tool()
async def qdrant_search(
    collection: str,
    query_vector: List[float],
    top_k: int = 5,
    qdrant_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    flt = Filter(**qdrant_filter) if qdrant_filter else None
    hits = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )
    return {
        "collection": collection,
        "top_k": top_k,
        "hits": [{"id": h.id, "score": h.score, "payload": h.payload} for h in hits],
    }

# ---- Atlassian tools (guarded) ----
@mcp.tool()
async def confluence_publish_page(
    title: str,
    body_html: str,
    space_key: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_atlassian_enabled()
    space = space_key or CONFLUENCE_SPACE_KEY
    if not space:
        raise RuntimeError("Missing Confluence space key (space_key or CONFLUENCE_SPACE_KEY)")

    parent = parent_id or CONFLUENCE_PARENT_ID or None

    payload: Dict[str, Any] = {
        "type": "page",
        "title": title,
        "space": {"key": space},
        "body": {"storage": {"value": body_html, "representation": "storage"}},
    }
    if parent:
        payload["ancestors"] = [{"id": parent}]

    async with httpx.AsyncClient(timeout=60, auth=_atlassian_auth()) as client:
        r = await client.post(f"{ATLASSIAN_BASE_URL}/wiki/rest/api/content", json=payload)
        r.raise_for_status()
        data = r.json()

    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "url": data.get("_links", {}).get("base", "") + data.get("_links", {}).get("webui", ""),
        "raw": data,
    }

@mcp.tool()
async def jira_create_issue(
    summary: str,
    description: str,
    issue_type: str,  # "Epic" | "Story" | "Task"
    project_key: Optional[str] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
    epic_name: Optional[str] = None,
) -> Dict[str, Any]:
    _ensure_atlassian_enabled()
    project = project_key or JIRA_PROJECT_KEY
    if not project:
        raise RuntimeError("Missing Jira project key (project_key or JIRA_PROJECT_KEY)")

    fields: Dict[str, Any] = {
        "project": {"key": project},
        "summary": summary,
        "issuetype": {"name": issue_type},
        "description": description,
    }

    if issue_type.lower() == "epic" and epic_name and JIRA_EPIC_NAME_FIELD:
        fields[JIRA_EPIC_NAME_FIELD] = epic_name

    if additional_fields:
        fields.update(additional_fields)

    async with httpx.AsyncClient(timeout=60, auth=_atlassian_auth()) as client:
        r = await client.post(f"{ATLASSIAN_BASE_URL}/rest/api/3/issue", json={"fields": fields})
        r.raise_for_status()
        data = r.json()

    key = data.get("key")
    return {
        "key": key,
        "id": data.get("id"),
        "browse_url": f"{ATLASSIAN_BASE_URL}/browse/{key}" if key else None,
        "raw": data,
    }

@mcp.tool()
async def jira_create_epic(summary: str, description: str, epic_name: str, project_key: Optional[str] = None) -> Dict[str, Any]:
    return await jira_create_issue(summary, description, "Epic", project_key=project_key, epic_name=epic_name)

@mcp.tool()
async def jira_create_story(
    summary: str,
    description: str,
    project_key: Optional[str] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await jira_create_issue(summary, description, "Story", project_key=project_key, additional_fields=additional_fields)

@mcp.tool()
async def jira_create_task(
    summary: str,
    description: str,
    project_key: Optional[str] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await jira_create_issue(summary, description, "Task", project_key=project_key, additional_fields=additional_fields)

# ---------------- Debug HTTP endpoints ----------------
async def health(_):
    return JSONResponse({"ok": True, "atlassian_enabled": ENABLE_ATLASSIAN_TOOLS})

async def tools(_):
    t = await mcp.list_tools()
    tool_list = t.tools if hasattr(t, "tools") else t
    return JSONResponse({"tools": [{"name": x.name, "description": x.description or ""} for x in tool_list]})

async def call_tool(request: Request):
    """
    Endpoint HTTP stable: exécute un tool MCP sans utiliser le client protocolaire.
    Body:
      { "tool": "ollama_generate", "args": { ... } }
    """
    payload = await request.json()
    tool = payload.get("tool")
    args = payload.get("args", {})

    if not tool:
        return JSONResponse({"error": "Missing 'tool' in body"}, status_code=400)

    try:
        result = await mcp.call_tool(tool, args)
        content = getattr(result, "content", result)
        return JSONResponse({"tool": tool, "content": content})
    except Exception as e:
        return JSONResponse({"tool": tool, "error": repr(e), "message": str(e)}, status_code=500)

routes = [
    Route("/health", health, methods=["GET"]),
    Route("/tools", tools, methods=["GET"]),
    Route("/call", call_tool, methods=["POST"]),
    Mount(MCP_MOUNT_PATH, app=mcp.streamable_http_app()),
]

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield

app = Starlette(routes=routes, lifespan=lifespan)

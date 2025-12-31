import os
import contextlib
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

from mcp.server.fastmcp import FastMCP

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

# Streamable HTTP MCP
mcp = FastMCP("agent-ia-mcp", stateless_http=True, json_response=True)


def _atlassian_auth() -> httpx.BasicAuth:
    # Atlassian Cloud: basic auth = email + API token
    return httpx.BasicAuth(ATLASSIAN_EMAIL, ATLASSIAN_API_TOKEN)


async def _ollama_post(path: str, payload: Dict[str, Any], timeout_s: float = 120) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(f"{OLLAMA_URL}{path}", json=payload)
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def ollama_generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    if options:
        payload["options"] = options
    data = await _ollama_post("/api/generate", payload, timeout_s=180)
    return {"model": model, "response": data.get("response", ""), "raw": data}


@mcp.tool()
async def ollama_embed(
    model: str,
    input: str,
) -> Dict[str, Any]:
    # Ollama embeddings endpoint
    data = await _ollama_post("/api/embeddings", {"model": model, "prompt": input}, timeout_s=60)
    # response typically contains: {"embedding":[...]}
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


# ---------- Atlassian tools (guardés) ----------

def _ensure_atlassian_enabled() -> None:
    if not ENABLE_ATLASSIAN_TOOLS:
        raise RuntimeError("Atlassian tools are disabled. Set ENABLE_ATLASSIAN_TOOLS=true")
    if not (ATLASSIAN_BASE_URL and ATLASSIAN_EMAIL and ATLASSIAN_API_TOKEN):
        raise RuntimeError("Missing ATLASSIAN_BASE_URL / ATLASSIAN_EMAIL / ATLASSIAN_API_TOKEN")


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

    # Epic: certains Jira exigent un champ custom "Epic Name"
    if issue_type.lower() == "epic" and epic_name and JIRA_EPIC_NAME_FIELD:
        fields[JIRA_EPIC_NAME_FIELD] = epic_name

    if additional_fields:
        fields.update(additional_fields)

    payload = {"fields": fields}

    async with httpx.AsyncClient(timeout=60, auth=_atlassian_auth()) as client:
        r = await client.post(f"{ATLASSIAN_BASE_URL}/rest/api/3/issue", json=payload)
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
async def jira_create_epic(
    summary: str,
    description: str,
    epic_name: str,
    project_key: Optional[str] = None,
) -> Dict[str, Any]:
    return await jira_create_issue(
        summary=summary,
        description=description,
        issue_type="Epic",
        project_key=project_key,
        epic_name=epic_name,
    )


@mcp.tool()
async def jira_create_story(
    summary: str,
    description: str,
    project_key: Optional[str] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await jira_create_issue(
        summary=summary,
        description=description,
        issue_type="Story",
        project_key=project_key,
        additional_fields=additional_fields,
    )


@mcp.tool()
async def jira_create_task(
    summary: str,
    description: str,
    project_key: Optional[str] = None,
    additional_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return await jira_create_issue(
        summary=summary,
        description=description,
        issue_type="Task",
        project_key=project_key,
        additional_fields=additional_fields,
    )


# ---------- Debug endpoints HTTP (pour Postman) ----------
async def health(_):
    return JSONResponse({"ok": True, "atlassian_enabled": ENABLE_ATLASSIAN_TOOLS})

async def tools(_):
    t = await mcp.list_tools()
    tool_list = t.tools if hasattr(t, "tools") else t
    return JSONResponse({"tools": [{"name": x.name, "description": x.description} for x in tool_list]})


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/tools", tools, methods=["GET"]),
    Mount(MCP_MOUNT_PATH, app=mcp.streamable_http_app()),
]

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with mcp.session_manager.run():
        yield

app = Starlette(routes=routes, lifespan=lifespan)

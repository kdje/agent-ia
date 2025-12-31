import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


from . import agent_common
from . import agent_architect
from . import agent_security
from . import agent_observability
from . import agent_data_contract
from . import agent_migration

import os
import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import traceback
import httpx


app = FastAPI(title="Integration Copilot API (multi-agents)", version="2.1.0")
MCP_URL = os.getenv("MCP_URL", "http://mcp:8001/mcp")


class DesignRequest(BaseModel):
    context: str
    systems: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    data_objects: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)

    # modèles optionnels par agent
    model_arch: Optional[str] = None
    model_sec: Optional[str] = None
    model_obs: Optional[str] = None
    model_dc: Optional[str] = None
    model_mig: Optional[str] = None

    # RAG contrôlé par l’API
    top_k: int = 8
    rag: bool = True

def _req_to_dict(req: DesignRequest) -> Dict[str, Any]:
    return {
        "context": req.context,
        "systems": req.systems,
        "constraints": req.constraints,
        "data_objects": req.data_objects,
        "triggers": req.triggers,
    }

# MCP
# MCP
async def _mcp_call(tool: str, args: Dict[str, Any]) -> Any:
    """
    Appelle un tool MCP via transport Streamable HTTP.
    Compatible avec les versions où streamablehttp_client() renvoie:
      - (read, write)
      - (read, write, close)
      - ou un objet avec .read/.write
    """
    async with streamablehttp_client(MCP_URL) as conn:
        # 1) cas tuple/list
        if isinstance(conn, (tuple, list)):
            if len(conn) < 2:
                raise RuntimeError(f"Unexpected MCP conn tuple length: {len(conn)}")
            read, write = conn[0], conn[1]
        else:
            # 2) cas objet
            read = getattr(conn, "read", None)
            write = getattr(conn, "write", None)
            if read is None or write is None:
                raise RuntimeError(f"Unexpected MCP conn type: {type(conn)}")

        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.call_tool(tool, args)
            return resp.content



@app.get("/health")
def health():
    return {
        "status": "ok",
        "qdrant_url": agent_common.QDRANT_URL,
        "collection": agent_common.COLLECTION,
        "embed_model": agent_common.EMBED_MODEL,
        "default_model": agent_common.DEFAULT_MODEL,
    }

@app.post("/architect")
def architect(req: DesignRequest):
    t0 = time.time()
    try:
        model = req.model_arch or agent_common.DEFAULT_MODEL
        r = agent_architect.RequirementInput.model_validate(_req_to_dict(req))
        out = agent_architect.generate(r, model=model, rag=req.rag, top_k=req.top_k)
        return {
            "agent": "architect",
            "model": model,
            "rag": req.rag,
            "top_k": req.top_k,
            "latency_ms": int((time.time() - t0) * 1000),
            "result": out.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/security")
def security(req: DesignRequest):
    t0 = time.time()
    try:
        model = req.model_sec or agent_common.DEFAULT_MODEL
        r = agent_security.RequirementInput.model_validate(_req_to_dict(req))
        out = agent_security.generate(r, model=model, rag=req.rag, top_k=req.top_k)
        return {
            "agent": "security",
            "model": model,
            "rag": req.rag,
            "top_k": req.top_k,
            "latency_ms": int((time.time() - t0) * 1000),
            "result": out.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/observability")
def observability(req: DesignRequest):
    t0 = time.time()
    try:
        model = req.model_obs or agent_common.DEFAULT_MODEL
        r = agent_observability.RequirementInput.model_validate(_req_to_dict(req))
        out = agent_observability.generate(r, model=model, rag=req.rag, top_k=req.top_k)
        return {
            "agent": "observability",
            "model": model,
            "rag": req.rag,
            "top_k": req.top_k,
            "latency_ms": int((time.time() - t0) * 1000),
            "result": out.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data-contract")
def data_contract(req: DesignRequest):
    t0 = time.time()
    try:
        model = req.model_dc or agent_common.DEFAULT_MODEL
        r = agent_data_contract.RequirementInput.model_validate(_req_to_dict(req))
        out = agent_data_contract.generate(r, model=model, rag=req.rag, top_k=req.top_k)
        return {
            "agent": "data_contract",
            "model": model,
            "rag": req.rag,
            "top_k": req.top_k,
            "latency_ms": int((time.time() - t0) * 1000),
            "result": out.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/migration")
def migration(req: DesignRequest):
    t0 = time.time()
    try:
        model = req.model_mig or agent_common.DEFAULT_MODEL
        r = agent_migration.RequirementInput.model_validate(_req_to_dict(req))
        out = agent_migration.generate(r, model=model, rag=req.rag, top_k=req.top_k)
        return {
            "agent": "migration",
            "model": model,
            "rag": req.rag,
            "top_k": req.top_k,
            "latency_ms": int((time.time() - t0) * 1000),
            "result": out.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/design")
def design(req: DesignRequest):
    """
    Lance tous les agents et renvoie un JSON consolidé.
    """
    t0 = time.time()
    try:
        data = _req_to_dict(req)

        model_arch = req.model_arch or "qwen2.5:14b-instruct"
        model_sec  = req.model_sec  or "mistral-nemo:latest"
        model_obs  = req.model_obs  or "mistral-nemo:latest"
        model_dc   = req.model_dc   or "qwen2.5:14b-instruct"
        model_mig  = req.model_mig  or "mistral-nemo:latest"

        arch_out = agent_architect.generate(
            agent_architect.RequirementInput.model_validate(data),
            model=model_arch, rag=req.rag, top_k=req.top_k
        )
        sec_out = agent_security.generate(
            agent_security.RequirementInput.model_validate(data),
            model=model_sec, rag=req.rag, top_k=req.top_k
        )
        obs_out = agent_observability.generate(
            agent_observability.RequirementInput.model_validate(data),
            model=model_obs, rag=req.rag, top_k=req.top_k
        )
        dc_out = agent_data_contract.generate(
            agent_data_contract.RequirementInput.model_validate(data),
            model=model_dc, rag=req.rag, top_k=req.top_k
        )
        mig_out = agent_migration.generate(
            agent_migration.RequirementInput.model_validate(data),
            model=model_mig, rag=req.rag, top_k=req.top_k
        )

        return {
            "models": {
                "architect": model_arch,
                "security": model_sec,
                "observability": model_obs,
                "data_contract": model_dc,
                "migration": model_mig,
            },
            "rag": req.rag,
            "top_k": req.top_k,
            "latency_ms": int((time.time() - t0) * 1000),
            "result": {
                "architect": arch_out.model_dump(),
                "security": sec_out.model_dump(),
                "observability": obs_out.model_dump(),
                "data_contract": dc_out.model_dump(),
                "migration": mig_out.model_dump(),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# POUR MCP

class ConfluencePublishRequest(BaseModel):
    title: str
    body_html: str
    space_key: Optional[str] = None
    parent_id: Optional[str] = None


class JiraIssueRequest(BaseModel):
    summary: str
    description: str
    project_key: Optional[str] = None
    additional_fields: Optional[Dict[str, Any]] = None


class JiraEpicRequest(JiraIssueRequest):
    epic_name: str


@app.post("/confluence/publish")
async def confluence_publish(req: ConfluencePublishRequest):
    """
    Publie une page Confluence via MCP.
    Le serveur MCP doit avoir ENABLE_ATLASSIAN_TOOLS=true.
    """
    try:
        return await _mcp_call("confluence_publish_page", req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jira/epic")
async def jira_create_epic(req: JiraEpicRequest):
    """
    Crée un Epic Jira via MCP.
    """
    try:
        return await _mcp_call("jira_create_epic", req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jira/story")
async def jira_create_story(req: JiraIssueRequest):
    """
    Crée une Story Jira via MCP.
    """
    try:
        return await _mcp_call("jira_create_story", req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jira/task")
async def jira_create_task(req: JiraIssueRequest):
    """
    Crée une Task Jira via MCP.
    """
    try:
        return await _mcp_call("jira_create_task", req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import traceback

@app.get("/mcp-health", operation_id="mcp_health_http")
async def mcp_health_http():
    base = MCP_URL.removesuffix("/mcp") if MCP_URL.endswith("/mcp") else MCP_URL
    async with httpx.AsyncClient(timeout=5.0) as client:
        h = await client.get(f"{base}/health")
        t = await client.get(f"{base}/tools")
    return {
        "ok": True,
        "mode": "http",
        "mcp_url": MCP_URL,
        "health": h.json(),
        "tools": t.json().get("tools", []),
    }


@app.get("/mcp-proto-health", operation_id="mcp_health_proto")
async def mcp_health_proto():
    try:
        async with streamablehttp_client(MCP_URL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool_list = tools.tools if hasattr(tools, "tools") else tools
                return {
                    "ok": True,
                    "mode": "mcp-proto",
                    "mcp_url": MCP_URL,
                    "tools": [t.name for t in tool_list],
                }
    except Exception as e:
        # AnyIO/ExceptionGroup => donne un message lisible
        if isinstance(e, ExceptionGroup):
            return HTTPException(status_code=500, detail={
                "mode": "mcp-proto",
                "error": "ExceptionGroup",
                "sub_errors": [str(x) for x in e.exceptions],
            })
        raise HTTPException(status_code=500, detail={"mode": "mcp-proto", "error": str(e)})

@app.get("/mcp-proto-ping", operation_id="mcp_proto_ping")
async def mcp_proto_ping():
    out = await _mcp_call("ollama_generate", {
        "model": "qwen2.5:14b-instruct",
        "prompt": "Réponds uniquement: pong"
    })
    return {"ok": True, "out": out}








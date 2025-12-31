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

app = FastAPI(title="Integration Copilot API (multi-agents)", version="2.1.0")

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

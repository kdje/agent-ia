import os
import time
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import copilot  # ✅ IMPORTANT (au lieu de "import copilot")

app = FastAPI(title="Integration Architect Copilot API", version="1.0.0")

class DesignRequest(BaseModel):
    context: str
    systems: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    data_objects: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    top_k: int = 8
    rag: bool = True

@app.get("/health")
def health():
    return {
        "status": "ok",
        "qdrant_url": copilot.QDRANT_URL,
        "collection": copilot.COLLECTION,
        "embed_model": copilot.EMBED_MODEL,
        "default_model": copilot.DEFAULT_MODEL,
    }

@app.post("/design")
def design(req: DesignRequest):
    t0 = time.time()
    model = req.model or copilot.DEFAULT_MODEL
    r = copilot.RequirementInput(
        context=req.context,
        systems=req.systems,
        constraints=req.constraints,
        data_objects=req.data_objects,
        triggers=req.triggers,
    )

    try:
        if req.rag:
            query = f"{r.context}\nSystems: {', '.join(r.systems)}\nData: {', '.join(r.data_objects)}"
            kb = copilot.retrieve_context(query, top_k=req.top_k)
        else:
            kb = []

        prompt = copilot.build_prompt(r, kb)
        content = copilot.call_llm(model, prompt)

        # parsing robuste (si tu as bien la version avec json_loads_lenient)
        data = copilot.json_loads_lenient(content)
        out = copilot.OutputDoc.model_validate(data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "model": model,
        "rag": req.rag,
        "top_k": req.top_k,
        "latency_ms": int((time.time() - t0) * 1000),
        "result": out.model_dump(),
    }

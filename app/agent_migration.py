import os, json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from .agent_common import DEFAULT_MODEL, retrieve_context, kb_block, call_llm, parse_json_with_repairs

class RequirementInput(BaseModel):
    context: str
    systems: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    data_objects: List[str] = Field(default_factory=list)
    triggers: List[str] = Field(default_factory=list)

class MigrationOutput(BaseModel):
    assumptions: List[str]
    open_questions: List[str]
    cutover: Dict[str, Any]
    rollback: Dict[str, Any]
    runbook: Dict[str, Any]
    risks: List[Dict[str, Any]]
    references: List[str]

SYSTEM_PROMPT = """Tu es Migration Agent (SAP S/4 RISE + MuleSoft).
Objectif: plan de cutover, rollback, et runbook opérationnel (pré-checks, validation, monitoring).
Réponds UNIQUEMENT en JSON valide conforme au schéma. Aucun texte hors JSON.
Si info manquante: hypothèses explicites + questions ouvertes (max 5).
"""

def build_prompt(req: RequirementInput, kb: List[Dict[str, str]], rag: bool, top_k: int) -> str:
    schema = MigrationOutput.model_json_schema()
    rag_block = kb_block(kb) if (rag and kb) else "(RAG désactivé ou aucun résultat)"
    return f"""
INPUT:
{req.model_dump_json(indent=2, ensure_ascii=False)}

RAG (rag={rag}, top_k={top_k}):
{rag_block}

SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

Règles:
- cutover: fenêtre, checklist, owners, critères GO/NO-GO
- rollback: critères déclencheurs + étapes + recovery
- runbook: pré-checks, commandes (génériques), validation fonctionnelle + technique, monitoring post-go-live
- risks: top risques + mitigations
- Si rag=false: references = []
- references = sources RAG réellement utilisées (pas inventées)
"""

def generate(req: RequirementInput, model: str = DEFAULT_MODEL, rag: bool = True, top_k: int = 8) -> MigrationOutput:
    kb: List[Dict[str, str]] = []
    if rag:
        rag_query = f"migration cutover rollback runbook SAP MuleSoft {req.context}"
        kb = retrieve_context(rag_query, top_k=top_k)

    prompt = build_prompt(req, kb, rag=rag, top_k=top_k)
    content = call_llm(model, SYSTEM_PROMPT, prompt)
    data = parse_json_with_repairs(model, SYSTEM_PROMPT, content, MigrationOutput.model_json_schema())
    return MigrationOutput.model_validate(data)

def main():
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="JSON file input (sinon stdin)")
    parser.add_argument("--model", default=os.getenv("COPILOT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    req_data = json.load(open(args.input, encoding="utf-8")) if args.input else json.load(sys.stdin)
    req = RequirementInput.model_validate(req_data)
    out = generate(req, model=args.model, rag=args.rag, top_k=args.top_k)
    print(out.model_dump_json(indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

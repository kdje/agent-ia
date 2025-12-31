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

class ArchOption(BaseModel):
    id: str
    title: str
    description: str
    pros: List[str]
    cons: List[str]
    risks: List[str]
    effort: str = Field(description="S|M|L")
    when_to_choose: str

class ArchitectOutput(BaseModel):
    assumptions: List[str]
    open_questions: List[str]
    options: List[ArchOption]
    recommendation: Dict[str, Any]  # {option_id, rationale, decision_drivers}
    adr: Dict[str, Any]             # {title,status,context,decision,consequences,alternatives}
    references: List[str]

SYSTEM_PROMPT = """Tu es Architect Agent (SAP S/4 RISE + MuleSoft).
Objectif: proposer 2-3 options d'architecture, recommander 1 option et produire un ADR.
Réponds UNIQUEMENT en JSON valide conforme au schéma. Aucun texte hors JSON.
Si info manquante: hypothèses explicites + questions ouvertes (max 5).
"""

def build_prompt(req: RequirementInput, kb: List[Dict[str, str]], rag: bool, top_k: int) -> str:
    schema = ArchitectOutput.model_json_schema()
    rag_block = kb_block(kb) if (rag and kb) else "(RAG désactivé ou aucun résultat)"
    return f"""
INPUT:
{req.model_dump_json(indent=2, ensure_ascii=False)}

RAG (rag={rag}, top_k={top_k}):
{rag_block}

SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

Règles:
- 2 à 3 options distinctes (id A/B/C)
- recommendation.option_id = un id existant
- adr.status = proposed
- Si rag=false: references = []
- references = liste des sources RAG réellement utilisées (pas inventées)
- Style: concret et actionnable (steps, risques, effort)
"""

def generate(req: RequirementInput, model: str = DEFAULT_MODEL, rag: bool = True, top_k: int = 8) -> ArchitectOutput:
    kb: List[Dict[str, str]] = []
    if rag:
        rag_query = f"architecture integration SAP S/4 RISE MuleSoft options ADR {req.context}"
        kb = retrieve_context(rag_query, top_k=top_k)

    prompt = build_prompt(req, kb, rag=rag, top_k=top_k)
    content = call_llm(model, SYSTEM_PROMPT, prompt)
    data = parse_json_with_repairs(model, SYSTEM_PROMPT, content, ArchitectOutput.model_json_schema())
    return ArchitectOutput.model_validate(data)

def main():
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="JSON file input (sinon stdin)")
    parser.add_argument("--model", default=os.getenv("COPILOT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction, default=True)  # --rag / --no-rag
    args = parser.parse_args()

    req_data = json.load(open(args.input, encoding="utf-8")) if args.input else json.load(sys.stdin)
    req = RequirementInput.model_validate(req_data)
    out = generate(req, model=args.model, rag=args.rag, top_k=args.top_k)
    print(out.model_dump_json(indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

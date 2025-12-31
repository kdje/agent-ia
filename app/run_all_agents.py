import os, json, sys
from typing import Any, Dict

from architect_agent import RequirementInput as ArchReq, generate as gen_arch
from security_agent import RequirementInput as SecReq, generate as gen_sec
from observability_agent import RequirementInput as ObsReq, generate as gen_obs
from data_contract_agent import RequirementInput as DcReq, generate as gen_dc
from migration_agent import RequirementInput as MigReq, generate as gen_mig

def _load_req(path: str | None) -> Dict[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.load(sys.stdin)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all Integration Copilot agents and output consolidated JSON.")
    parser.add_argument("-i", "--input", help="Input JSON file (otherwise read from stdin)")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rag", action=argparse.BooleanOptionalAction, default=True)  # --rag / --no-rag
    parser.add_argument("--model-arch", default=os.getenv("COPILOT_MODEL_ARCH", os.getenv("COPILOT_MODEL", "qwen2.5:14b-instruct")))
    parser.add_argument("--model-sec",  default=os.getenv("COPILOT_MODEL_SEC",  os.getenv("COPILOT_MODEL", "mistral-nemo:latest")))
    parser.add_argument("--model-obs",  default=os.getenv("COPILOT_MODEL_OBS",  os.getenv("COPILOT_MODEL", "mistral-nemo:latest")))
    parser.add_argument("--model-dc",   default=os.getenv("COPILOT_MODEL_DC",   os.getenv("COPILOT_MODEL", "qwen2.5:14b-instruct")))
    parser.add_argument("--model-mig",  default=os.getenv("COPILOT_MODEL_MIG",  os.getenv("COPILOT_MODEL", "mistral-nemo:latest")))
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    req_data = _load_req(args.input)

    arch_req = ArchReq.model_validate(req_data)
    sec_req  = SecReq.model_validate(req_data)
    obs_req  = ObsReq.model_validate(req_data)
    dc_req   = DcReq.model_validate(req_data)
    mig_req  = MigReq.model_validate(req_data)

    result = {
        "architect": gen_arch(arch_req, model=args.model_arch, rag=args.rag, top_k=args.top_k).model_dump(),
        "security": gen_sec(sec_req,  model=args.model_sec,  rag=args.rag, top_k=args.top_k).model_dump(),
        "observability": gen_obs(obs_req, model=args.model_obs, rag=args.rag, top_k=args.top_k).model_dump(),
        "data_contract": gen_dc(dc_req, model=args.model_dc, rag=args.rag, top_k=args.top_k).model_dump(),
        "migration": gen_mig(mig_req, model=args.model_mig, rag=args.rag, top_k=args.top_k).model_dump(),
    }

    if args.pretty:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()

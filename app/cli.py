from typing import Optional
import typer

from copilot import RequirementInput, generate, DEFAULT_MODEL

app = typer.Typer(help="Integration Architect Copilot CLI")

@app.command()
def design(
    # contexte possible en argument positionnel...
    context: Optional[str] = typer.Argument(None, help="Contexte (si --context n'est pas fourni)"),
    # ...ou en option --context
    context_opt: Optional[str] = typer.Option(None, "--context", help="Contexte du besoin"),
    systems: str = typer.Option("", "--systems", help="Systèmes séparés par virgules"),
    constraints: str = typer.Option("", "--constraints", help="Contraintes séparées par ;"),
    data_objects: str = typer.Option("", "--data-objects", help="Objets data séparés par virgules"),
    triggers: str = typer.Option("", "--triggers", help="Triggers séparés par virgules"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Modèle Ollama (Qwen/Nemo)"),
):
    ctx = context_opt or context
    if not ctx:
        raise typer.BadParameter("Fournis un contexte via --context ou en argument positionnel.")

    req = RequirementInput(
        context=ctx,
        systems=[s.strip() for s in systems.split(",") if s.strip()],
        constraints=[s.strip() for s in constraints.split(";") if s.strip()],
        data_objects=[s.strip() for s in data_objects.split(",") if s.strip()],
        triggers=[s.strip() for s in triggers.split(",") if s.strip()],
    )

    out = generate(req, model=model)
    print(out.model_dump_json(indent=2, ensure_ascii=False))

if __name__ == "__main__":
    app()

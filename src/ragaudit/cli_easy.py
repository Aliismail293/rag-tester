"""Bodies of the first-run commands: `init`, `doctor` and `check`.

Registered in cli.py like the v2 commands in cli_v2.py.
"""

from pathlib import Path

import typer

from ragaudit.scaffold import ENV_EXAMPLE_NAME, FRAMEWORKS, write_starter_files


def _fail(message: str) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


# --- init --------------------------------------------------------------------


def run_init(framework: str, out: Path, force: bool) -> list[Path]:
    if framework not in FRAMEWORKS:
        _fail(f"--framework must be one of {', '.join(FRAMEWORKS)}, got {framework!r}")
    try:
        written = write_starter_files(framework, out, force=force)
    except FileExistsError as exc:
        _fail(str(exc))
    env_example = out.parent / ENV_EXAMPLE_NAME
    env_file = out.parent / ".env"
    typer.echo(f"Wrote {out} ({framework} adapter template) and {env_example}.")
    typer.echo("Next:")
    typer.echo(f"  1. cp {env_example} {env_file}   and fill in your API settings (OpenAI-compatible or local Ollama).")
    typer.echo(f"  2. Open {out} and connect your pipeline at the TODO.")
    typer.echo(f"  3. ragaudit doctor --adapter {out}")
    typer.echo(f"  4. ragaudit check --docs ./your-docs --adapter {out}")
    return written

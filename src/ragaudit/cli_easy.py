"""Bodies of the first-run commands: `init`, `doctor` and `check`.

Registered in cli.py like the v2 commands in cli_v2.py.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import typer
from pydantic import ValidationError

from ragaudit.adapters.base import AdapterLoadError, load_adapter
from ragaudit.llm.client import LLMClient, LLMClientError
from ragaudit.llm.factory import PROVIDER_ENV, resolve_provider
from ragaudit.models.adapter import RAGResponse
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


# --- doctor ------------------------------------------------------------------


MIN_PYTHON = (3, 11)
API_ENV_VARS = ("RAGAUDIT_API_BASE", "RAGAUDIT_API_KEY", "RAGAUDIT_MODEL")
DOCTOR_QUESTION = "What is this document about?"
ENV_FIX = "add it to .env (`cp .env.example .env`, from `ragaudit init`) or export it in your shell."


@dataclass
class Check:
    name: str
    ok: bool
    fix: str | None = None


def check_python(version: tuple = tuple(sys.version_info[:3])) -> Check:
    shown = ".".join(str(part) for part in version)
    if tuple(version[:2]) >= MIN_PYTHON:
        return Check(f"Python {shown} (3.11+ required)", True)
    return Check(
        f"Python {shown} is too old (3.11+ required)",
        False,
        "install Python 3.11 or newer, then `python3.11 -m pip install ragaudit` in a fresh virtualenv.",
    )


def check_env_vars(mock: bool) -> list[Check]:
    checks = []
    examples = {
        "RAGAUDIT_API_BASE": "RAGAUDIT_API_BASE=https://api.openai.com/v1 (or http://localhost:11434/v1 for Ollama)",
        "RAGAUDIT_API_KEY": "RAGAUDIT_API_KEY=sk-... (or RAGAUDIT_API_KEY=ollama for Ollama)",
        "RAGAUDIT_MODEL": "RAGAUDIT_MODEL=gpt-4o-mini (or an Ollama model such as llama3.1:8b)",
    }
    for name in API_ENV_VARS:
        if os.environ.get(name):
            checks.append(Check(f"{name} is set", True))
        elif mock:
            checks.append(Check(f"{name} is not set (not needed with the mock LLM)", True))
        else:
            checks.append(Check(f"{name} is not set", False, f"set {examples[name]}: {ENV_FIX}"))
    return checks


def _is_local_ollama(api_base: str) -> bool:
    return ":11434" in api_base


def _api_fix(error: str, api_base: str) -> str:
    ollama = _is_local_ollama(api_base)
    if "status 401" in error or "status 403" in error:
        if ollama:
            return "the server refused the key. For Ollama set RAGAUDIT_API_KEY=ollama."
        return "the server refused RAGAUDIT_API_KEY: check the key is valid and has access to RAGAUDIT_MODEL."
    if "status 404" in error:
        if ollama:
            return (
                "model or URL not found. Run `ollama list` to see your models (`ollama pull llama3.1:8b` to get one), "
                "set RAGAUDIT_MODEL to one of them, and make sure RAGAUDIT_API_BASE ends in /v1."
            )
        return "model or URL not found: check RAGAUDIT_MODEL is a model your provider serves and RAGAUDIT_API_BASE ends in /v1."
    if "status 429" in error:
        return "rate limited or out of quota: check your provider's billing and limits, then retry."
    if "status" in error:
        return "the API returned an error: check RAGAUDIT_API_BASE points at an OpenAI-compatible /v1 endpoint."
    if ollama:
        return "could not reach Ollama: start it with `ollama serve` (or open the Ollama app), then retry."
    return "could not reach RAGAUDIT_API_BASE: check the URL (it should end in /v1) and your network."


def check_api(mock: bool, env_ok: bool, client_factory=None) -> Check:
    name = "Test call to the LLM API"
    if mock:
        return Check(f"{name}: skipped, mock LLM selected ({PROVIDER_ENV}=mock); its scores are not meaningful", True)
    if not env_ok:
        return Check(f"{name}: not attempted", False, "set the RAGAUDIT_* variables above first.")
    api_base = os.environ.get("RAGAUDIT_API_BASE", "")
    if not api_base.startswith(("http://", "https://")):
        return Check(f"{name}: RAGAUDIT_API_BASE is not a URL", False, "it must start with http:// or https:// and end in /v1.")
    try:
        client = (client_factory or LLMClient)(max_retries=1, timeout=30.0)
        reply = client.complete("You are a health check. Reply with the single word OK.", "Are you there?")
    except LLMClientError as exc:
        return Check(f"{name} failed", False, _api_fix(str(exc), api_base))
    except Exception as exc:  # malformed responses, bad URLs
        return Check(f"{name} failed ({type(exc).__name__})", False, _api_fix(str(exc), api_base))
    if not str(reply).strip():
        return Check(f"{name}: the model replied with nothing", False, "check RAGAUDIT_MODEL is a chat model.")
    return Check(f"{name} answered", True)


def check_adapter_import(adapter_path: Path):
    name = f"Adapter {adapter_path} imports"
    if not adapter_path.is_file():
        return Check(f"Adapter {adapter_path} not found", False, "run `ragaudit init`, or pass --adapter PATH."), None
    try:
        adapter = load_adapter(str(adapter_path))
    except AdapterLoadError as exc:
        cause = exc.__cause__
        detail = f"{type(cause).__name__}: {cause}" if cause is not None else str(exc)
        if isinstance(cause, ModuleNotFoundError):
            fix = f"install the missing package ({detail}) in this environment, e.g. `pip install {cause.name}`."
        elif cause is not None:
            fix = f"fix the error raised while importing it ({detail})."
        else:
            fix = "define `query(question) -> RAGResponse` (or an `adapter` object). `ragaudit init` writes a template."
        return Check(f"Adapter {adapter_path} does not import", False, fix), None
    return Check(name, True), adapter


def check_adapter_response(adapter, adapter_path: Path, question: str = DOCTOR_QUESTION) -> list[Check]:
    name = "Adapter returns a valid RAGResponse"
    if adapter is None:
        return [Check(f"{name}: not attempted", False, "fix the adapter import above first.")]
    try:
        response = adapter.query(question)
    except NotImplementedError:
        return [Check(f"{name}: the template is not filled in", False, f"connect your pipeline at the TODO in {adapter_path}.")]
    except Exception as exc:
        return [Check(f"{name}: query() raised {type(exc).__name__}: {exc}", False, f"fix query() in {adapter_path}; it must not raise on a normal question.")]
    if isinstance(response, dict):
        try:
            response = RAGResponse.model_validate(response)
        except ValidationError:
            pass
    if not isinstance(response, RAGResponse):
        return [Check(
            f"{name}: query() returned {type(response).__name__}",
            False,
            "return RAGResponse(answer=..., contexts=[...]) (from ragaudit.models.adapter import RAGResponse).",
        )]
    if not response.answer.strip():
        return [Check(f"{name}: the answer is empty", False, f"return your pipeline's answer text in `answer` in {adapter_path}.")]
    checks = [Check(f"{name} on a sample question", True)]
    if response.contexts:
        checks.append(Check(f"Adapter returns retrieved passages ({len(response.contexts)} in `contexts`)", True))
    else:
        checks.append(Check(
            "Adapter returns no retrieved passages (`contexts` is empty)",
            False,
            "return the text of each retrieved passage in `contexts`; `ragaudit check` needs them to score "
            "retrieval (or use --match-by id if you return ragaudit chunk ids).",
        ))
    return checks


def run_doctor(adapter_path: Path, llm: str | None = None) -> list[Check]:
    try:
        mock = resolve_provider(llm) == "mock"
    except ValueError as exc:
        _fail(str(exc))
    typer.echo("ragaudit doctor")
    if Path(".env").is_file():
        typer.echo("  (read settings from ./.env; variables exported in your shell take priority)")
    checks = [check_python()]
    env_checks = check_env_vars(mock)
    checks += env_checks
    checks.append(check_api(mock, all(c.ok for c in env_checks)))
    import_check, adapter = check_adapter_import(adapter_path)
    checks.append(import_check)
    checks += check_adapter_response(adapter, adapter_path)

    for check in checks:
        if check.ok:
            typer.secho(f"  PASS  {check.name}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"  FAIL  {check.name}", fg=typer.colors.RED)
            typer.echo(f"        Fix: {check.fix}")
    failed = sum(1 for check in checks if not check.ok)
    if failed:
        typer.secho(f"{failed} check(s) failed. Fix them and run `ragaudit doctor` again.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho(
        f"All checks passed. Next: ragaudit check --docs ./your-docs --adapter {adapter_path}", fg=typer.colors.GREEN
    )
    return checks

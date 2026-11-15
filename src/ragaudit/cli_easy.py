"""Bodies of the first-run commands: `init`, `doctor` and `check`.

Registered in cli.py like the v2 commands in cli_v2.py.
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import typer
from pydantic import ValidationError

from ragaudit.adapters.base import AdapterLoadError, load_adapter
from ragaudit.llm.client import LLMClient, LLMClientError
from ragaudit.llm.factory import PROVIDER_ENV, resolve_provider
from ragaudit.models.adapter import RAGResponse
from ragaudit.scaffold import DEFAULT_DEMO_DOCS, DEMO_DOCS_ENV, ENV_EXAMPLE_NAME, FRAMEWORKS, write_starter_files


def _fail(message: str) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@contextmanager
def _adapter_env(llm: str | None, demo_docs: Path | None = None):
    """Expose --llm (and check's --docs) to adapters that call ragaudit's LLM, like the demo one.

    Sets RAGAUDIT_LLM only when --llm was passed, and restores the
    environment afterwards.
    """
    updates = {}
    if llm:
        updates[PROVIDER_ENV] = llm
    if demo_docs is not None:
        updates[DEMO_DOCS_ENV] = str(demo_docs.resolve())
    saved = {name: os.environ.get(name) for name in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def adapter_llm_calls(adapter) -> int:
    """LLM calls the adapter itself makes per question through ragaudit's LLM (the demo adapter: 1)."""
    fn = getattr(adapter, "_fn", None)  # a query() function wrapped by CallableAdapter
    flagged = getattr(adapter, "USES_RAGAUDIT_LLM", False) or getattr(fn, "__globals__", {}).get("USES_RAGAUDIT_LLM", False)
    return 1 if flagged else 0


# --- init --------------------------------------------------------------------


def run_init(framework: str, out: Path, force: bool, docs: Path = DEFAULT_DEMO_DOCS) -> list[Path]:
    if framework not in FRAMEWORKS:
        _fail(f"--framework must be one of {', '.join(FRAMEWORKS)}, got {framework!r}")
    try:
        written = write_starter_files(framework, out, force=force, docs=docs)
    except FileExistsError as exc:
        _fail(str(exc))
    env_example = out.parent / ENV_EXAMPLE_NAME
    env_file = out.parent / ".env"
    if framework == "demo":
        typer.echo(f"Wrote {out} (a ready-to-run demo pipeline: keyword search over {docs}) and {env_example}.")
        if not docs.is_dir():
            typer.echo(f"  Note: {docs} does not exist yet; put a few .txt or .md files in it.")
        typer.echo("Next:")
        typer.echo(f"  1. cp {env_example} {env_file}   and fill in your API settings (OpenAI-compatible or local Ollama).")
        typer.echo(f"  2. ragaudit check --docs {docs} --adapter {out}")
        typer.echo("     (add --llm mock to try it offline with a fake LLM first)")
        return written
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
        "install Python 3.11 or newer, then `python3.11 -m pip install ragaudit-cli` in a fresh virtualenv.",
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
    with _adapter_env(llm):
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


# --- check -------------------------------------------------------------------

DEFAULT_CHECK_QUESTIONS = 20
MAX_CHECK_QUESTIONS = 500
DEFAULT_CHECK_DB = Path(".ragaudit") / "check.db"
DEFAULT_CHECK_REPORT = Path("ragaudit-report.html")
JUDGE_CALLS_PER_QUESTION = 2  # correctness + groundedness
ABLATION_CALLS_PER_QUESTION = 2  # no-context answer + its correctness judgment, for lucky passes only


@dataclass
class CheckPlan:
    chunks: list
    """The chunks questions will be written from."""
    per_chunk: list[int]
    """Questions to ask of each chunk, in the same order."""

    @property
    def questions(self) -> int:
        return sum(self.per_chunk)

    @property
    def generate_calls(self) -> int:
        return len(self.chunks)

    @property
    def expected_calls(self) -> int:
        return self.generate_calls + JUDGE_CALLS_PER_QUESTION * self.questions

    @property
    def max_calls(self) -> int:
        return self.expected_calls + ABLATION_CALLS_PER_QUESTION * self.questions


def plan_check(chunks: list, questions: int) -> CheckPlan:
    """Spread `questions` evenly over the corpus: one each from evenly spaced chunks, or more per chunk when there are few."""
    if questions <= len(chunks):
        selected = [chunks[(i * len(chunks)) // questions] for i in range(questions)]
        return CheckPlan(selected, [1] * questions)
    base, extra = divmod(questions, len(chunks))
    return CheckPlan(list(chunks), [base + 1 if i < extra else base for i in range(len(chunks))])


def print_summary_box(lines: list[str]) -> None:
    typer.echo("")
    typer.secho("ragaudit check: summary", bold=True)
    for line in lines:
        typer.echo(f"  {line}")


def run_check(
    docs: Path,
    adapter_path: Path,
    questions: int,
    match_by: str,
    match_threshold: float,
    yes: bool,
    no_open: bool,
    max_llm_calls: int | None,
    db: Path,
    out: Path,
    llm: str | None,
    note: str | None = None,
    fresh_questions: bool = False,
    legacy: bool = False,
    progress=None,
) -> list[str]:
    """ingest -> generate (reusing earlier questions) -> run (with ablation) -> history -> report, in one command."""
    with _adapter_env(llm, docs if docs.is_dir() else None):
        return _run_check(
            docs, adapter_path, questions, match_by, match_threshold, yes, no_open, max_llm_calls, db, out, llm,
            note, fresh_questions, legacy, progress,
        )


def stable_chunks(docs: Path, documents: list) -> list[tuple]:
    """(stable-id chunk, v1 chunk) pairs: the same text under v2's `relative/path.md#position` id and v1's content-hash id.

    The pool keys questions by the stable id, so an edit elsewhere never
    orphans them; the run itself keeps v1 ids so --match-by id behaves as before.
    """
    from ragaudit.ingest import chunker

    pairs = []
    for document in documents:
        relative = Path(document.source_path).relative_to(docs).as_posix()
        for chunk in chunker.chunk_document(document):
            stable = chunk.model_copy(update={"chunk_id": f"{relative}#{chunk.position}", "doc_id": relative})
            pairs.append((stable, chunk))
    return pairs


def _describe_plan(plan) -> str:
    if plan.fresh:
        return f"Questions: writing a new set of {plan.new_questions} ({plan.fresh_reason})."
    text = f"Questions: reusing {len(plan.reused)} from earlier checks, so scores stay comparable"
    if plan.new_questions:
        counts = plan.counts
        text += (
            f"; writing {plan.new_questions} for new or edited passages "
            f"({counts['new']} new, {counts['changed']} edited, {counts['deleted']} deleted)"
        )
    if plan.retire:
        text += f"; retiring {len(plan.retire)} from edited or deleted passages"
    return text + ". (--fresh-questions writes a brand-new set.)"


def _run_check(
    docs: Path,
    adapter_path: Path,
    questions: int,
    match_by: str,
    match_threshold: float,
    yes: bool,
    no_open: bool,
    max_llm_calls: int | None,
    db: Path,
    out: Path,
    llm: str | None,
    note: str | None,
    fresh_questions: bool,
    legacy: bool,
    progress,
) -> list[str]:
    import webbrowser

    from ragaudit import check_questions, history
    from ragaudit.cli import _execute_run
    from ragaudit.generate.testset_generator import generate_test_cases_for_chunk, generate_test_set
    from ragaudit.ingest.loader import load_documents
    from ragaudit.ingest.versioning import compute_index_version
    from ragaudit.llm.factory import BudgetedCompleter, CallBudget, LLMBudgetExceeded, make_client
    from ragaudit.report.plain_summary import build_plain_summary
    from ragaudit.scoring.text_match import validate_match_by
    from ragaudit.storage import db as db_module
    from ragaudit.storage import repository
    from ragaudit.storage.store import Store

    def report_progress(phase: str, done: int, total: int, calls: int = 0) -> None:
        if progress is not None:
            progress(phase, done, total, calls)

    try:
        validate_match_by(match_by)
        provider = resolve_provider(llm)
    except ValueError as exc:
        _fail(str(exc))
    if not 1 <= questions <= MAX_CHECK_QUESTIONS:
        _fail(f"--questions must be between 1 and {MAX_CHECK_QUESTIONS}, got {questions}")
    if max_llm_calls is not None and max_llm_calls < 1:
        _fail("--max-llm-calls must be at least 1")
    if not docs.exists():
        _fail(f"docs folder not found: {docs}")
    documents = load_documents(docs)
    if not documents:
        _fail(f"no .txt or .md files found under {docs}")
    pairs = stable_chunks(docs, documents)
    if not pairs:
        _fail(f"the files under {docs} are empty")
    chunks = [v1 for _, v1 in pairs]
    stable = [s for s, _ in pairs]
    v1_by_stable = {s.chunk_id: v1 for s, v1 in pairs}
    try:
        adapter = load_adapter(str(adapter_path))
    except AdapterLoadError as exc:
        _fail(f"{exc} Run `ragaudit doctor --adapter {adapter_path}` for the exact fix.")
    # One sample question before any LLM spend, so a broken adapter costs nothing.
    problems = [c for c in check_adapter_response(adapter, adapter_path) if not c.ok]
    if match_by == "id":  # empty contexts only block text matching
        problems = [c for c in problems if "`contexts`" not in c.name]
    if problems:
        _fail(f"{problems[0].name}. Fix: {problems[0].fix}")

    store_path = history.store_path_for_docs(docs)
    store = Store(store_path)
    try:
        plan = check_questions.plan_questions(store, stable, questions, fresh=fresh_questions)
        estimate = CheckPlan([chunk for chunk, _ in plan.to_generate], [n for _, n in plan.to_generate])
        n_total = plan.questions
        judge_calls = JUDGE_CALLS_PER_QUESTION * n_total
        expected = estimate.generate_calls + judge_calls
        upper = expected + ABLATION_CALLS_PER_QUESTION * n_total
        cap = max_llm_calls if max_llm_calls is not None else upper
        typer.echo(f"Docs: {len(documents)} file(s), {len(chunks)} chunk(s). Questions: {n_total}, from {estimate.generate_calls} chunk(s)." if plan.fresh else f"Docs: {len(documents)} file(s), {len(chunks)} chunk(s). Questions: {n_total}.")
        typer.echo(_describe_plan(plan))
        typer.echo(
            f"Estimated LLM calls: about {expected} ({estimate.generate_calls} to write questions, "
            f"{judge_calls} to judge answers), up to {upper} if every answer needs "
            f"the lucky-pass check. Budget cap: {cap} call(s) (--max-llm-calls)."
        )
        own_calls = adapter_llm_calls(adapter)
        if own_calls:
            typer.echo(
                f"The demo adapter answers through the same LLM: {own_calls * n_total} more call(s), outside the cap "
                f"(1 was made just now to test it)."
            )
        else:
            typer.echo(f"Your adapter will be called {n_total} more time(s); its own LLM calls are not counted here.")
        if provider == "mock":
            typer.echo("LLM: mock (offline, free; its scores say nothing about a real pipeline).")
        if not yes and not typer.confirm("Continue?", default=True):
            if own_calls:
                typer.echo("Cancelled. Only the adapter's one test call was sent to the LLM.")
            else:
                typer.echo("Cancelled. Nothing was sent to the LLM.")
            raise typer.Exit(code=0)

        budget = CallBudget(cap)
        client = BudgetedCompleter(make_client(provider), budget)

        db.parent.mkdir(parents=True, exist_ok=True)
        if db.exists():
            db.unlink()  # the working database starts clean; questions and history live in the project store
        conn = db_module.get_connection(db)
        phase = "writing questions"
        try:
            repository.save_documents(conn, documents)
            repository.save_chunks(conn, chunks)
            check_questions.commit_plan(store, plan, stable, questions)
            pooled = list(plan.reused)
            if plan.to_generate:
                typer.echo(f"Writing {plan.new_questions} question(s) from your docs...")
            for index, (chunk, n) in enumerate(plan.to_generate, start=1):
                written = generate_test_cases_for_chunk(client, chunk, n=n)[:n]
                pooled.extend(check_questions.add_generated(store, chunk, written))
                report_progress("generate", index, len(plan.to_generate), budget.used)
                typer.echo(f"  generate {index}/{len(plan.to_generate)}")
            seen, unique = set(), []
            for q in pooled:
                if q.key not in seen and q.chunk_id in v1_by_stable:
                    seen.add(q.key)
                    unique.append(q)
            cases = [check_questions.to_test_case(q, v1_by_stable[q.chunk_id]) for q in unique]
            test_set = generate_test_set(client, documents, [], n_per_chunk=1)
            test_set = test_set.model_copy(update={"test_cases": cases})
            if not cases:
                _fail("the LLM did not return any usable questions; check RAGAUDIT_MODEL with `ragaudit doctor`.")
            if len(cases) < n_total:
                typer.echo(f"Got {len(cases)} of {n_total} question(s); some passages gave fewer than asked.")
            repository.save_test_set(conn, test_set)
            phase = "scoring answers"
            run = _execute_run(
                conn, test_set, adapter_path, client,
                run_ablation_flag=True, match_by=match_by, match_threshold=match_threshold,
                on_case=(lambda done, total: report_progress("answer", done, total, budget.used)),
            )
        except LLMBudgetExceeded:
            _fail(
                f"stopped while {phase}: the budget of {cap} LLM call(s) was used up. "
                f"Raise --max-llm-calls or lower --questions (now {questions}). Questions written so far are saved "
                "and reused next time."
            )
        finally:
            conn.close()

        corpus_fingerprint = compute_index_version({s.chunk_id: chunk_hash_of(s) for s in stable})
        row = history.record_run(
            store,
            kind="check",
            run_id=run.config.run_id,
            outcomes={case.test_id: case.verdict.value for case in run.scored_cases},
            adapter_path=str(adapter_path),
            corpus_fingerprint=corpus_fingerprint,
            correct_count=sum(case.correctness.is_correct for case in run.scored_cases),
            retrieval_hits=sum(case.retrieval.hit for case in run.scored_cases),
            groundedness_pct=round(run.summary.mean_groundedness * 100, 1) if run.summary.mean_groundedness is not None else None,
            models={"judge": client.model, "questions": test_set.generation_model},
            note=note,
            corpus_changed=plan.corpus_changed,
            extra={
                "llm_calls": budget.used,
                "match_by": match_by,
                "match_threshold": match_threshold,
                "docs": str(docs),
                "reused_questions": len(plan.reused),
                "new_questions": len(cases) - len(plan.reused),
            },
        )
        rows = history.load_history(store)
    finally:
        store.close()

    report_path = render_run_report(
        run, test_set, out, legacy=legacy, history_rows=rows, docs=docs, llm_calls=budget.used,
        chunk_texts={c.chunk_id: c.text for c in chunks}, project_store=store_path,
    )
    typer.echo(f"Used {budget.used} LLM call(s). Report: {report_path}")
    if row.prev_run_id is not None:
        if row.shared_questions:
            typer.echo(
                f"vs the previous check: {row.delta_true_pass:+.1f} true-pass points on the "
                f"{row.shared_questions} question(s) both runs share."
            )
        else:
            typer.echo("vs the previous check: no shared questions, so no like-for-like delta.")
    if not no_open:
        webbrowser.open(report_path.resolve().as_uri())

    lines = build_plain_summary(run.summary, run.scored_cases).lines()
    print_summary_box(lines)
    return lines


def chunk_hash_of(chunk) -> str:
    from ragaudit.ingest.corpus import chunk_hash

    return chunk_hash(chunk)


def render_run_report(run, test_set, out: Path, *, legacy: bool = False, **kwargs) -> Path:
    """The run report: the v3 report by default, the v1 template with legacy=True."""
    from ragaudit.report.generator import generate_report

    if legacy:
        return generate_report(run, test_set, out)
    try:
        from ragaudit.report.v3 import render_v1_run
    except ImportError:  # the v3 report lands in a later commit
        return generate_report(run, test_set, out)
    return render_v1_run(run, test_set, out, **kwargs)

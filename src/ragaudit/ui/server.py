"""`ragaudit ui`: a local setup page, so the terminal part is two commands.

A stdlib `ThreadingHTTPServer` bound to 127.0.0.1 only. Security:

- every request needs the random session token (the `X-Ragaudit-Token`
  header, or `?t=` for page loads); state-changing POSTs need the header,
  which another website can't set without a CORS preflight we never allow;
- the Host header must be this server (stops DNS rebinding), and any
  Origin / Sec-Fetch-Site header must be same-origin;
- every path the page sends is resolved inside the project folder;
- the server calls ragaudit functions directly, never a shell, and never
  sends an API key back to the page.

`check` and `improve` run as one background job at a time; the page polls
`/api/job` every second and can cancel it.
"""

import io
import json
import os
import secrets
import sys
import threading
import time
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULT_PORT = 8765
OLLAMA_URL = "http://localhost:11434"
OLLAMA_API_BASE = "http://localhost:11434/v1"
RECOMMENDED_MODELS = {"qwen2.5:7b": "about 4.7 GB", "llama3.1:8b": "about 4.9 GB"}
"""Writes and grades the questions (first) and checks them (second, a different model)."""
QUICK_QUESTIONS = 20
DEFAULT_IMPROVE_BUDGET = 400
ADAPTER_NAME = "ragaudit_adapter.py"
REPORT_NAME = "ragaudit-report.html"
ENV_KEYS = ("RAGAUDIT_API_BASE", "RAGAUDIT_API_KEY", "RAGAUDIT_MODEL", "RAGAUDIT_VERIFIER_MODEL")
FRAMEWORKS = {"demo": "Demo (no code)", "plain": "Plain Python", "langchain": "LangChain", "llamaindex": "LlamaIndex"}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class JobCancelled(BaseException):  # not an Exception, so no `except Exception` on the way swallows it
    pass


class _ThreadRouter(io.TextIOBase):
    """A sys.stdout/sys.stderr stand-in that sends a job thread's output to its log, everything else through."""

    def __init__(self, real):
        self.real = real
        self.sinks: dict[int, "Job"] = {}

    def write(self, text: str) -> int:
        job = self.sinks.get(threading.get_ident())
        if job is not None:
            job.write(text)
        else:
            self.real.write(text)
        return len(text)

    def flush(self) -> None:
        self.real.flush()

    def isatty(self) -> bool:
        return False


class Job:
    def __init__(self, kind: str):
        self.kind = kind
        self.state = "running"
        self.phase = "starting"
        self.done = 0
        self.total = 0
        self.calls = 0
        self.lines: list[str] = []
        self._partial = ""
        self.error: str | None = None
        self.message: str | None = None
        self.report: str | None = None
        self.cancel = threading.Event()
        self.started = time.time()
        self.finished: float | None = None
        self.lock = threading.Lock()

    def write(self, text: str) -> None:
        with self.lock:
            self._partial += text
            *complete, self._partial = self._partial.split("\n")
            self.lines.extend(line.rstrip() for line in complete if line.strip())
            self.lines = self.lines[-200:]

    def progress(self, phase: str, done: int, total: int, calls: int) -> None:
        if self.cancel.is_set():
            raise JobCancelled()
        with self.lock:
            self.phase, self.done, self.total, self.calls = phase, done, total, calls

    def as_dict(self) -> dict:
        with self.lock:
            return {
                "kind": self.kind,
                "state": self.state,
                "phase": self.phase,
                "done": self.done,
                "total": self.total,
                "calls": self.calls,
                "log": self.lines[-40:],
                "error": self.error,
                "message": self.message,
                "has_report": self.report is not None,
                "elapsed": round((self.finished or time.time()) - self.started),
            }


# --- .env --------------------------------------------------------------------


def write_env(path: Path, values: dict[str, str | None]) -> None:
    """Set RAGAUDIT_* keys in a .env file, keeping every other line. None removes a key. File mode 600."""
    lines = path.read_text().splitlines() if path.is_file() else [
        "# ragaudit settings, written by `ragaudit ui`. Keep this file out of git."
    ]
    remaining = dict(values)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in remaining:
            value = remaining.pop(key)
            if value is not None:
                out.append(f"{key}={value}")
            continue
        out.append(line)
    out.extend(f"{key}={value}" for key, value in remaining.items() if value is not None)
    path.write_text("\n".join(out) + "\n")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def read_env(path: Path) -> dict[str, str]:
    from ragaudit.env import parse_dotenv

    return parse_dotenv(path.read_text()) if path.is_file() else {}


# --- the app -------------------------------------------------------------------


class SetupApp:
    """Everything the page can ask for, confined to one project folder."""

    def __init__(self, project: Path, llm: str | None = None):
        self.project = project.resolve()
        self.llm = llm
        self.job: Job | None = None
        self.job_lock = threading.Lock()
        self.env_path = self.project / ".env"
        self.state_path = self.project / ".ragaudit" / "ui.json"
        # The page's settings win over anything inherited, but only for this process.
        for key, value in read_env(self.env_path).items():
            if key in ENV_KEYS and key not in os.environ:
                os.environ[key] = value

    # --- helpers

    def inside(self, raw: str, what: str = "path") -> Path:
        raw = (raw or "").strip()
        if not raw:
            raise ApiError(f"enter a {what}")
        candidate = Path(raw).expanduser()
        resolved = (candidate if candidate.is_absolute() else self.project / candidate).resolve()
        if resolved != self.project and not resolved.is_relative_to(self.project):
            raise ApiError(f"the {what} must be inside the project folder {self.project}")
        return resolved

    def rel(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.project).as_posix()
        return "./" + relative if relative != "." else "."

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def save_state(self, **updates) -> dict:
        state = {**self.load_state(), **updates}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))
        return state

    @property
    def mock(self) -> bool:
        from ragaudit.llm.factory import resolve_provider

        try:
            return resolve_provider(self.llm) == "mock"
        except ValueError:
            return False

    def settings(self) -> dict:
        model = os.environ.get("RAGAUDIT_MODEL")
        base = os.environ.get("RAGAUDIT_API_BASE", "")
        return {
            "mock": self.mock,
            "provider": "mock" if self.mock else ("ollama" if ":11434" in base else ("key" if model else None)),
            "model": model,
            "verifier_model": os.environ.get("RAGAUDIT_VERIFIER_MODEL"),
            "api_base": base or None,
            "key_saved": bool(os.environ.get("RAGAUDIT_API_KEY")),
            "ready": self.mock or bool(model and base and os.environ.get("RAGAUDIT_API_KEY")),
        }

    def docs_path(self) -> Path | None:
        docs = self.load_state().get("docs")
        if not docs:
            return None
        try:
            return self.inside(docs, "docs folder")
        except ApiError:
            return None

    def adapter_path(self) -> Path:
        return self.project / ADAPTER_NAME

    # --- endpoints

    def state(self) -> dict:
        saved = self.load_state()
        settings = self.settings()
        docs = self.docs_path()
        adapter = self.adapter_path()
        docs_ok = bool(docs and docs.is_dir())
        steps_done = [settings["ready"], docs_ok, adapter.is_file()]
        report = self.project / REPORT_NAME
        return {
            "project": str(self.project),
            "settings": settings,
            "docs": self.rel(docs) if docs else "./docs",
            "docs_ok": docs_ok,
            "adapter": ADAPTER_NAME,
            "adapter_exists": adapter.is_file(),
            "framework": saved.get("framework", "demo"),
            "steps_done": steps_done,
            "start_step": 3 if all(steps_done) else 0,
            "has_report": report.is_file(),
            "job": self.job.as_dict() if self.job else None,
            "frameworks": FRAMEWORKS,
            "recommended": RECOMMENDED_MODELS,
            "quick_questions": QUICK_QUESTIONS,
            "improve_budget": DEFAULT_IMPROVE_BUDGET,
        }

    def ollama(self) -> dict:
        import httpx

        try:
            response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=1.5)
            response.raise_for_status()
            names = sorted(m.get("name", "") for m in response.json().get("models", []))
        except Exception:
            return {"found": False, "models": [], "missing": list(RECOMMENDED_MODELS), "sizes": RECOMMENDED_MODELS}
        present = set(names) | {n.removesuffix(":latest") for n in names}
        missing = [m for m in RECOMMENDED_MODELS if m not in present]
        return {"found": True, "models": names, "missing": missing, "sizes": RECOMMENDED_MODELS}

    def save_settings(self, body: dict) -> dict:
        provider = body.get("provider")
        if self.mock:
            return {"ok": True, "settings": self.settings()}
        if provider == "ollama":
            info = self.ollama()
            if not info["found"]:
                raise ApiError("Ollama is not running. Start the Ollama app (or `ollama serve`) and try again.")
            models = [m.removesuffix(":latest") for m in info["models"]]
            if not models:
                raise ApiError("Ollama has no models yet. Use Download models first.")
            generator = next((m for m in RECOMMENDED_MODELS if m in models), models[0])
            verifier = next((m for m in RECOMMENDED_MODELS if m in models and m != generator), None)
            verifier = verifier or next((m for m in models if m != generator), None)
            values = {
                "RAGAUDIT_API_BASE": OLLAMA_API_BASE,
                "RAGAUDIT_API_KEY": "ollama",
                "RAGAUDIT_MODEL": generator,
                "RAGAUDIT_VERIFIER_MODEL": verifier,
            }
        elif provider == "key":
            base = (body.get("api_base") or "").strip()
            model = (body.get("model") or "").strip()
            key = (body.get("api_key") or "").strip() or os.environ.get("RAGAUDIT_API_KEY")
            if not base.startswith(("http://", "https://")):
                raise ApiError("the base URL must start with http:// or https:// (for OpenAI: https://api.openai.com/v1)")
            if not model:
                raise ApiError("enter a model name, for example gpt-4o-mini")
            if not key:
                raise ApiError("paste your API key")
            verifier = (body.get("verifier_model") or "").strip() or None
            if verifier == model:
                raise ApiError("the second model must be different from the first, so it can catch its mistakes")
            values = {"RAGAUDIT_API_BASE": base, "RAGAUDIT_API_KEY": key, "RAGAUDIT_MODEL": model, "RAGAUDIT_VERIFIER_MODEL": verifier}
        else:
            raise ApiError("pick Ollama or an API key")
        write_env(self.env_path, values)
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return {"ok": True, "settings": self.settings()}

    def test_key(self, body: dict) -> dict:
        from ragaudit.cli_easy import _api_fix
        from ragaudit.llm.client import LLMClient

        if self.mock:
            return {"ok": True, "message": "Mock LLM: nothing to test."}
        base = (body.get("api_base") or os.environ.get("RAGAUDIT_API_BASE") or "").strip()
        model = (body.get("model") or os.environ.get("RAGAUDIT_MODEL") or "").strip()
        key = (body.get("api_key") or "").strip() or os.environ.get("RAGAUDIT_API_KEY") or ""
        if not base.startswith(("http://", "https://")) or not model or not key:
            raise ApiError("fill in the key, the base URL and the model first")
        try:
            client = LLMClient(api_base=base, api_key=key, model=model, max_retries=1, timeout=30.0)
            reply = client.complete("You are a health check. Reply with the single word OK.", "Are you there?")
        except Exception as exc:  # never echo the key: messages come from status codes and the server's text
            message = str(exc).replace(key, "***") if key else str(exc)
            return {"ok": False, "message": "The test call failed.", "fix": _api_fix(message, base)}
        if not str(reply).strip():
            return {"ok": False, "message": "The model replied with nothing.", "fix": "check the model is a chat model."}
        return {"ok": True, "message": f"{model} answered."}

    def docs(self, body: dict) -> dict:
        from ragaudit.ingest.corpus import CorpusError, load_corpus
        from ragaudit.ingest.loader import SUPPORTED_EXTENSIONS

        path = self.inside(body.get("path") or "./docs", "docs folder")
        if not path.is_dir():
            return {"ok": False, "path": self.rel(path) if path.exists() or path.parent.exists() else body.get("path"),
                    "message": "That folder does not exist yet. Create it and put your .txt or .md files in it."}
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
        skipped = sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() not in SUPPORTED_EXTENSIONS and not p.name.startswith("."))
        try:
            passages = len(load_corpus(path)) if files else 0
        except CorpusError:
            passages = 0
        result = {
            "ok": bool(files) and passages > 0,
            "path": self.rel(path),
            "count": len(files),
            "names": [p.relative_to(path).as_posix() for p in files[:12]],
            "more": max(0, len(files) - 12),
            "passages": passages,
            "skipped": skipped,
        }
        if not files:
            result["message"] = "No .txt or .md files in this folder yet."
        elif not passages:
            result["message"] = "The files are empty."
        if body.get("save") and result["ok"]:
            self.save_state(docs=result["path"])
        return result

    def adapter_preview(self, body: dict) -> dict:
        from ragaudit.scaffold import adapter_template

        framework = body.get("framework") or "demo"
        if framework not in FRAMEWORKS:
            raise ApiError("unknown framework")
        docs = self.docs_path() or self.project / "docs"
        code = adapter_template(framework, ADAPTER_NAME, os.path.relpath(docs, self.project))
        existing = self.adapter_path()
        return {
            "code": code,
            "exists": existing.is_file(),
            "same": existing.is_file() and existing.read_text() == code,
            "name": ADAPTER_NAME,
        }

    def write_adapter(self, body: dict) -> dict:
        from ragaudit.scaffold import ENV_EXAMPLE, ENV_EXAMPLE_NAME, adapter_template

        framework = body.get("framework") or "demo"
        if framework not in FRAMEWORKS:
            raise ApiError("unknown framework")
        path = self.adapter_path()
        if path.is_file() and not body.get("overwrite"):
            return {"ok": False, "needs_confirm": True, "message": f"{ADAPTER_NAME} already exists. Replace it?"}
        docs = self.docs_path() or self.project / "docs"
        path.write_text(adapter_template(framework, ADAPTER_NAME, os.path.relpath(docs, self.project)))
        env_example = self.project / ENV_EXAMPLE_NAME
        if not env_example.exists():
            env_example.write_text(ENV_EXAMPLE)
        self.save_state(framework=framework)
        edit = "" if framework == "demo" else " Open it, connect your app at the TODO, then test the connection."
        return {"ok": True, "message": f"Wrote {ADAPTER_NAME}.{edit}"}

    def test_connection(self, body: dict) -> dict:
        from ragaudit.cli_easy import _adapter_env, check_adapter_import, check_adapter_response

        path = self.adapter_path()
        docs = self.docs_path()
        with _adapter_env("mock" if self.mock else None, docs if docs and docs.is_dir() else None):
            import_check, adapter = check_adapter_import(path)
            checks = [import_check] + (check_adapter_response(adapter, Path(ADAPTER_NAME)) if adapter is not None else [])
        failed = next((c for c in checks if not c.ok), None)
        if failed is not None:
            return {"ok": False, "message": failed.name, "fix": failed.fix}
        count = next((c.name.split("(")[1].split(" ")[0] for c in checks if "retrieved passages (" in c.name), None)
        return {"ok": True, "message": "Answered a sample question" + (f" and returned {count} passages." if count else ".")}

    # --- jobs

    def _start(self, kind: str, target) -> dict:
        with self.job_lock:
            if self.job is not None and self.job.state == "running":
                raise ApiError("something is already running; wait for it or cancel it", 409)
            job = Job(kind)
            self.job = job
        thread = threading.Thread(target=self._run_job, args=(job, target), daemon=True)
        thread.start()
        return job.as_dict()

    def _run_job(self, job: Job, target) -> None:
        import typer

        install_output_router()  # idempotent; also re-installs if something swapped sys.stdout since serve()
        routers = [s for s in (sys.stdout, sys.stderr) if isinstance(s, _ThreadRouter)]
        for router in routers:
            router.sinks[threading.get_ident()] = job
        try:
            target(job)
            job.state = "done"
        except JobCancelled:
            job.state = "cancelled"
            job.message = "Cancelled. Anything finished so far is saved."
        except typer.Exit as exc:
            if getattr(exc, "exit_code", 0) == 0:
                job.state = "stopped"
                job.message = job.lines[-1] if job.lines else "Stopped."
            else:
                job.state = "error"
                job.error = next((l[len("Error: "):] for l in reversed(job.lines) if l.startswith("Error: ")), "it stopped with an error")
        except Exception as exc:
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.write(traceback.format_exc())
        finally:
            job.finished = time.time()
            for router in routers:
                router.sinks.pop(threading.get_ident(), None)

    def run(self, body: dict) -> dict:
        from ragaudit.cli_easy import run_check
        from ragaudit.cli_improve import run_improve
        from ragaudit.scoring.text_match import DEFAULT_MATCH_THRESHOLD

        mode = body.get("mode")
        docs = self.docs_path()
        adapter = self.adapter_path()
        if not self.settings()["ready"]:
            raise ApiError("finish step 1 (pick the AI) first")
        if docs is None or not docs.is_dir():
            raise ApiError("finish step 2 (your documents) first")
        if not adapter.is_file():
            raise ApiError("finish step 3 (connect your app) first")
        try:
            budget = int(body.get("budget") or 0) or None
        except (TypeError, ValueError):
            raise ApiError("the budget must be a whole number of AI calls")
        if budget is not None and budget < 1:
            raise ApiError("the budget must be at least 1")
        note = (body.get("note") or "").strip() or None
        llm = "mock" if self.mock else None
        out = self.project / REPORT_NAME

        if mode == "check":
            def target(job: Job) -> None:
                run_check(
                    docs, adapter, QUICK_QUESTIONS, "text", DEFAULT_MATCH_THRESHOLD, True, True, budget,
                    self.project / ".ragaudit" / "check.db", out, llm, note=note, progress=job.progress,
                )
                job.report = str(out)
        elif mode == "improve":
            def target(job: Job) -> None:
                run_improve(
                    docs, adapter, budget_calls=budget or DEFAULT_IMPROVE_BUDGET, yes=True, no_open=True, llm=llm,
                    note=note, out=out, config_out=self.project / "best_config.yaml", progress=job.progress,
                )
                job.report = str(out)
        else:
            raise ApiError("mode must be check or improve")
        return self._start(mode, target)

    def pull(self, body: dict) -> dict:
        import httpx

        models = body.get("models") or []
        if not models or any(m not in RECOMMENDED_MODELS for m in models):
            raise ApiError(f"only these models can be downloaded here: {', '.join(RECOMMENDED_MODELS)}")

        def target(job: Job) -> None:
            for index, model in enumerate(models, start=1):
                job.write(f"Downloading {model} ({index} of {len(models)})...\n")
                with httpx.stream("POST", f"{OLLAMA_URL}/api/pull", json={"model": model, "stream": True}, timeout=None) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        event = json.loads(line)
                        if event.get("error"):
                            raise RuntimeError(event["error"])
                        if event.get("total"):
                            job.progress(f"pull {model}", int(event.get("completed", 0)), int(event["total"]), index)
                        elif job.cancel.is_set():
                            raise JobCancelled()
                job.write(f"{model} is ready.\n")

        return self._start("pull", target)

    def cancel(self) -> dict:
        if self.job is not None and self.job.state == "running":
            self.job.cancel.set()
        return {"ok": True}

    def report_file(self) -> Path | None:
        if self.job is not None and self.job.report and Path(self.job.report).is_file():
            return Path(self.job.report)
        path = self.project / REPORT_NAME
        return path if path.is_file() else None


# --- HTTP ----------------------------------------------------------------------


def _page() -> str:
    return resources.files("ragaudit.ui").joinpath("setup.html").read_text(encoding="utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "ragaudit-ui"
    app: SetupApp
    token: str
    port: int

    def log_message(self, format, *args):  # noqa: A002 - never log URLs: they carry the token
        return

    # --- security

    def _allowed_hosts(self) -> set[str]:
        return {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def _check(self, require_header: bool) -> str | None:
        host = self.headers.get("Host", "")
        if host not in self._allowed_hosts():
            return "bad host"
        origin = self.headers.get("Origin")
        if origin is not None and origin.removeprefix("http://") not in self._allowed_hosts():
            return "cross-origin request refused"
        site = self.headers.get("Sec-Fetch-Site")
        navigation = self.command == "GET" and self.headers.get("Sec-Fetch-Mode") == "navigate" and not urlparse(self.path).path.startswith("/api/")
        # Opening the printed link from another page is a navigation: allowed (it still needs the token,
        # and the other page can't read the response). API calls must come from this page itself.
        if site is not None and site not in ("same-origin", "none") and not navigation:
            return "cross-site request refused"
        sent = self.headers.get("X-Ragaudit-Token")
        if sent is None and not require_header:
            sent = (parse_qs(urlparse(self.path).query).get("t") or [None])[0]
        if not sent or not secrets.compare_digest(sent, self.server.token):
            return "missing or wrong session token"
        return None

    # --- responses

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "SAMEORIGIN")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json")

    def _refuse(self, reason: str) -> None:
        self._json(HTTPStatus.FORBIDDEN, {"error": reason})

    # --- routes

    def do_GET(self) -> None:  # noqa: N802
        problem = self._check(require_header=False)
        if problem:
            return self._refuse(problem)
        app = self.server.app
        path = urlparse(self.path).path
        try:
            if path == "/":
                return self._send(200, _page().encode("utf-8"), "text/html; charset=utf-8")
            if path == "/report":
                report = app.report_file()
                if report is None:
                    return self._json(404, {"error": "no report yet"})
                return self._send(200, report.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(200, app.state())
            if path == "/api/ollama":
                return self._json(200, app.ollama())
            if path == "/api/job":
                return self._json(200, {"job": app.job.as_dict() if app.job else None})
        except ApiError as exc:
            return self._json(exc.status, {"error": str(exc)})
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        problem = self._check(require_header=True)
        if problem:
            return self._refuse(problem)
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._json(415, {"error": "send JSON"})
        try:
            length = min(int(self.headers.get("Content-Length") or 0), 1_000_000)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._json(400, {"error": "bad JSON"})
        app = self.server.app
        routes = {
            "/api/settings": app.save_settings,
            "/api/test-key": app.test_key,
            "/api/docs": app.docs,
            "/api/adapter/preview": app.adapter_preview,
            "/api/adapter": app.write_adapter,
            "/api/test-connection": app.test_connection,
            "/api/run": app.run,
            "/api/ollama/pull": app.pull,
            "/api/job/cancel": lambda _body: app.cancel(),
        }
        handler = routes.get(urlparse(self.path).path)
        if handler is None:
            return self._json(404, {"error": "not found"})
        try:
            return self._json(200, handler(body))
        except ApiError as exc:
            return self._json(exc.status, {"error": str(exc)})
        except Exception as exc:
            return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_PUT(self) -> None:  # noqa: N802
        self._json(405, {"error": "method not allowed"})

    do_DELETE = do_PUT


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, project: Path, port: int = DEFAULT_PORT, llm: str | None = None):
        try:
            super().__init__(("127.0.0.1", port), Handler)
        except OSError:
            if port == 0:
                raise
            super().__init__(("127.0.0.1", 0), Handler)  # the usual port is busy: take any free one
        self.app = SetupApp(project, llm)
        self.token = secrets.token_urlsafe(24)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}/?t={self.token}"


def install_output_router() -> None:
    if not isinstance(sys.stdout, _ThreadRouter):
        sys.stdout = _ThreadRouter(sys.stdout)
    if not isinstance(sys.stderr, _ThreadRouter):
        sys.stderr = _ThreadRouter(sys.stderr)


def serve(project: Path, port: int = DEFAULT_PORT, llm: str | None = None, open_browser: bool = True) -> None:
    import typer

    server = UIServer(project, port, llm)
    install_output_router()
    typer.echo(f"ragaudit setup page: {server.url}")
    typer.echo(f"Project folder: {server.app.project}")
    typer.echo("Running on this computer only. Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("\nStopped.")
    finally:
        server.server_close()

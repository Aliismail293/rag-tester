"""How `tune` picks the next config: grid, seeded random, or an LLM reading dev-set failures.

Every strategy only ever sees dev-split information: trial configs, dev
scores, dev failure-class counts and a few failed dev examples. Proposals
are validated against the declared search space; an LLM proposal that is
outside it (or already tried) gets one corrective retry, then a seeded
random config is used instead and the trial is marked `llm_fallback`.
"""

import itertools
import json
import random
import re
from ragaudit.llm.factory import Completer
from ragaudit.models.tune import Trial
from ragaudit.tune.search_space import SearchSpace, SearchSpaceError, canonical

STATE_MARKER = "STATE_JSON:"

PROPOSER_SYSTEM_PROMPT = (
    "You are tuning a RAG pipeline. You will get the search space (every knob and its allowed values), the "
    "configs tried so far with their dev-set scores (pass rate in points: the gold passage was retrieved AND "
    "the answer was correct), the failure-class counts of the best config so far, and a few failed dev "
    "examples. Failure classes: retrieval_miss = the gold passage was not in the top k; generation_miss = it "
    "was retrieved but the answer was still wrong.\n"
    "\n"
    "Propose ONE new config, not yet tried, that is likely to score higher. Every knob must be set, and every "
    "value must be one of that knob's allowed values exactly. Reason from the failures: retrieval misses call "
    "for retrieval knobs, generation misses for generation knobs.\n"
    "\n"
    "Respond with strict JSON only, no prose, no markdown fences, in the form: "
    '{"config": {knob: value, ...}, "rationale": str}'
)

RETRIEVAL_KNOB_ORDER = ["top_k", "hybrid_weight", "reranker", "chunk_size", "chunk_overlap"]
GENERATION_KNOB_ORDER = ["prompt_variant", "chunk_size", "chunk_overlap", "top_k", "reranker"]


class Proposed:
    """A strategy's next config, where it came from, and (for llm) why."""

    def __init__(self, config: dict, source: str, rationale: str | None = None) -> None:
        self.config = config
        self.source = source
        self.rationale = rationale


def _tried(trials: list[Trial]) -> set[str]:
    return {canonical(t.config) for t in trials}


class GridStrategy:
    name = "grid"

    def __init__(self, space: SearchSpace) -> None:
        self._grid = space.grid()

    def propose(self, trials: list[Trial], feedback: dict) -> Proposed | None:
        tried = _tried(trials)
        for config in self._grid:
            if canonical(config) not in tried:
                return Proposed(config, "grid")
        return None


class RandomStrategy:
    name = "random"

    def __init__(self, space: SearchSpace, seed: int) -> None:
        self.space = space
        self.rng = random.Random(seed)
        self._order: list[dict] | None = None

    def propose(self, trials: list[Trial], feedback: dict) -> Proposed | None:
        tried = _tried(trials)
        if self.space.size <= 100_000:
            if self._order is None:
                self._order = list(self.space.grid())
                self.rng.shuffle(self._order)
            for config in self._order:
                if canonical(config) not in tried:
                    return Proposed(config, "random")
            return None
        for _ in range(1000):
            config = self.space.random_config(self.rng)
            if canonical(config) not in tried:
                return Proposed(config, "random")
        return None


def build_proposer_prompt(space: SearchSpace, trials: list[Trial], feedback: dict) -> str:
    state = {
        "search_space": space.knobs,
        "trials": [
            {"config": t.config, "dev_score": t.dev.score, "failure_counts": t.dev.failure_counts} for t in trials
        ],
        "best_config": feedback.get("best_config"),
        "best_dev_score": feedback.get("best_dev_score"),
        "failure_counts": feedback.get("failure_counts", {}),
        "failed_dev_examples": feedback.get("failed_examples", []),
    }
    return f"{STATE_MARKER}\n{json.dumps(state, indent=1, default=str)}"


def parse_proposal(raw: str) -> tuple[dict, str | None] | None:
    text = raw.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    try:
        data = json.loads(fenced.group(1) if fenced else text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
        return None
    rationale = data.get("rationale")
    return data["config"], rationale if isinstance(rationale, str) else None


class LLMStrategy:
    name = "llm"

    def __init__(self, space: SearchSpace, client: Completer, seed: int) -> None:
        self.space = space
        self.client = client
        self._fallback = RandomStrategy(space, seed)

    def propose(self, trials: list[Trial], feedback: dict) -> Proposed | None:
        tried = _tried(trials)
        if len(tried) >= self.space.size:
            return None
        prompt = build_proposer_prompt(self.space, trials, feedback)
        problem = None
        for _ in range(2):
            user = prompt if problem is None else f"{prompt}\n\nYour previous proposal was rejected: {problem}. Try again."
            parsed = parse_proposal(self.client.complete(PROPOSER_SYSTEM_PROMPT, user))
            if parsed is None:
                problem = "it was not valid JSON of the form {\"config\": {...}, \"rationale\": str}"
                continue
            config, rationale = parsed
            try:
                config = self.space.normalize(config)
            except SearchSpaceError as exc:
                problem = str(exc)
                continue
            if canonical(config) in tried:
                problem = f"config {canonical(config)} was already tried"
                continue
            return Proposed(config, "llm", rationale)

        fallback = self._fallback.propose(trials, feedback)
        if fallback is None:
            return None
        return Proposed(fallback.config, "llm_fallback", f"LLM proposal rejected ({problem}); used a random config")


def make_strategy(name: str, space: SearchSpace, seed: int, client: Completer | None = None):
    if name == "grid":
        return GridStrategy(space)
    if name == "random":
        return RandomStrategy(space, seed)
    if name == "llm":
        if client is None:
            raise ValueError("the llm strategy needs an LLM client")
        return LLMStrategy(space, client, seed)
    raise ValueError(f"unknown strategy {name!r}; expected grid, random or llm")


def mock_proposal(user: str) -> dict:
    """What MockLLM answers to the proposer prompt: coordinate ascent steered by the dominant failure class.

    Starting from the best config, change one knob, picking the knob list
    for whichever failure class dominated the most recent trial (the
    latest evidence), and prefer the next value up before going down.
    Deterministic.
    """
    state = json.loads(user.split(STATE_MARKER, 1)[1].split("\n\nYour previous proposal", 1)[0])
    space: dict[str, list] = state["search_space"]
    tried = {canonical(t["config"]) for t in state["trials"]}
    best = state["best_config"] or {knob: values[0] for knob, values in space.items()}
    failures = (state["trials"][-1].get("failure_counts") if state["trials"] else None) or state.get("failure_counts") or {}
    retrieval_heavy = failures.get("retrieval_miss", 0) >= failures.get("generation_miss", 0)
    preferred = RETRIEVAL_KNOB_ORDER if retrieval_heavy else GENERATION_KNOB_ORDER
    order = [k for k in preferred if k in space] + [k for k in space if k not in preferred]
    dominant = "retrieval misses" if retrieval_heavy else "generation misses"

    for knob in order:
        values = space[knob]
        position = values.index(best[knob]) if best[knob] in values else 0
        for candidate in values[position + 1 :] + list(reversed(values[:position])):
            config = dict(best, **{knob: candidate})
            if canonical(config) not in tried:
                return {"config": config, "rationale": f"{dominant} dominate, so try {knob}={candidate}"}

    for combo in itertools.product(*space.values()):
        config = dict(zip(space, combo))
        if canonical(config) not in tried:
            return {"config": config, "rationale": "every single-knob change was tried; next untried config"}
    return {"config": best, "rationale": "search space exhausted"}

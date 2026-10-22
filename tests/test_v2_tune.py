import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragaudit import yaml_io
from ragaudit.cli import app
from ragaudit.llm.mock import MockLLM
from ragaudit.models.adapter import RAGResponse
from ragaudit.models.question import Question, Split
from ragaudit.storage import lock
from ragaudit.storage.store import Store
from ragaudit.tune import loop as tune_loop
from ragaudit.tune.loop import TuneError, run_tune
from ragaudit.tune.search_space import SearchSpaceError, parse_search_space
from ragaudit.tune.strategies import (
    PROPOSER_SYSTEM_PROMPT,
    GridStrategy,
    LLMStrategy,
    RandomStrategy,
    mock_proposal,
)

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "v2"


def make_questions(n: int) -> list[Question]:
    return [
        Question.create(
            chunk_ids=[f"c{i}"],
            chunk_hashes=[f"h{i}"],
            fact=f"fact {i}",
            question=f"question {i}?",
            gold_answer=f"gold {i}",
            generator_model="gen",
        )
        for i in range(n)
    ]


class QualityAdapter:
    """Always retrieves the gold chunk; answers correctly for questions in `good`."""

    def __init__(self, good: set[int]):
        self.good = good

    def query(self, question: str) -> RAGResponse:
        n = int(question.split()[1].rstrip("?"))
        return RAGResponse(answer=f"gold {n}" if n in self.good else "no idea", retrieved_chunk_ids=[f"c{n}"])


def quality_factory(questions: list[Question], overfit: bool = False):
    """quality 0..2 answers 0%, 50%, 100% correctly. With overfit, quality 2 is only right on dev questions."""
    numbers = {q.question_id: int(q.question.split()[1].rstrip("?")) for q in questions}
    dev_numbers = {numbers[q.question_id] for q in questions if q.split == Split.DEV}
    all_numbers = set(numbers.values())

    def factory(config: dict):
        quality = config["quality"]
        if quality == 0:
            return QualityAdapter(set())
        if quality == 1:
            return QualityAdapter({n for n in all_numbers if n % 2 == 0})
        return QualityAdapter(dev_numbers if overfit else all_numbers)

    return factory


SPACE = parse_search_space({"knobs": {"quality": [0, 1, 2], "flavor": ["a", "b"]}})


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "store.db") as s:
        yield s


def tune(store, questions, **kwargs):
    defaults = dict(strategy="grid", max_trials=20)
    defaults.update(kwargs)
    overfit = defaults.pop("overfit", False)
    space = defaults.pop("space", SPACE)
    return run_tune(
        store, quality_factory(questions, overfit), "adapter-sha", "fake.py", space, MockLLM("mock-judge"), "index-1",
        **defaults,
    )


# --- search space ---


def test_search_space_expands_ranges_and_defaults_the_baseline():
    space = parse_search_space(
        {"knobs": {"top_k": [1, 3], "w": {"min": 0.0, "max": 1.0, "step": 0.25}}, "baseline": {"top_k": 3}}
    )
    assert space.knobs["w"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert space.baseline == {"top_k": 3, "w": 0.0}
    assert space.size == 10


@pytest.mark.parametrize(
    "data, message",
    [
        ({"knobs": {}}, "non-empty"),
        ({"knobs": {"k": [1]}, "baseline": {"k": 2}}, "outside the search space"),
        ({"knobs": {"k": [1]}, "baseline": {"other": 1}}, "unknown knob"),
        ({"knobs": {"w": {"min": 0, "max": 1}}}, "min, max and step"),
        ({"knobs": {"w": {"min": 1, "max": 0, "step": 1}}}, "max >= min"),
    ],
)
def test_bad_search_spaces_are_rejected(data, message):
    with pytest.raises(SearchSpaceError, match=message):
        parse_search_space(data)


def test_normalize_accepts_equivalent_numbers_and_rejects_out_of_space():
    space = parse_search_space({"knobs": {"w": [0.0, 0.5], "k": [1, 3]}})
    assert space.normalize({"w": 0, "k": 3.0}) == {"w": 0.0, "k": 3}
    assert not space.contains({"w": 0.25, "k": 1})
    assert not space.contains({"w": 0.0})


# --- strategies ---


def test_grid_strategy_walks_the_grid_in_order_without_repeats():
    strategy = GridStrategy(SPACE)
    proposals = []
    trials = []
    for _ in range(SPACE.size + 1):
        proposal = strategy.propose(trials, {})
        if proposal is None:
            break
        proposals.append(proposal.config)
        trials.append(type("T", (), {"config": proposal.config})())
    assert len(proposals) == SPACE.size
    assert proposals[0] == {"quality": 0, "flavor": "a"}


def test_random_strategy_is_seeded_and_never_repeats():
    def sequence(seed):
        strategy, trials, configs = RandomStrategy(SPACE, seed=seed), [], []
        while (proposal := strategy.propose(trials, {})) is not None:
            configs.append(proposal.config)
            trials.append(type("T", (), {"config": proposal.config})())
        return configs

    first = sequence(5)
    assert first == sequence(5)
    assert len(first) == SPACE.size == len({json.dumps(c, sort_keys=True) for c in first})
    assert first != sequence(6)


class ScriptedProposer:
    def __init__(self, responses):
        self.model = "proposer"
        self.responses = list(responses)
        self.prompts = []

    def complete(self, system, user):
        assert system == PROPOSER_SYSTEM_PROMPT
        self.prompts.append(user)
        return self.responses.pop(0)


def test_llm_strategy_accepts_an_in_space_proposal():
    proposer = ScriptedProposer([json.dumps({"config": {"quality": 2, "flavor": "b"}, "rationale": "why not"})])
    proposal = LLMStrategy(SPACE, proposer, seed=0).propose([], {})
    assert proposal.config == {"quality": 2, "flavor": "b"} and proposal.source == "llm"


def test_llm_strategy_retries_once_then_falls_back_to_random():
    proposer = ScriptedProposer(
        [json.dumps({"config": {"quality": 9, "flavor": "b"}}), "not json"]
    )
    proposal = LLMStrategy(SPACE, proposer, seed=0).propose([], {})
    assert proposal.source == "llm_fallback"
    assert SPACE.contains(proposal.config)
    assert "rejected" in proposer.prompts[1]


def test_mock_proposer_steers_by_the_dominant_failure_class():
    space = {"top_k": [1, 3], "prompt_variant": ["lead", "extractive"]}
    best = {"top_k": 1, "prompt_variant": "lead"}

    def state(failures):
        payload = {"search_space": space, "trials": [{"config": best, "dev_score": 10, "failure_counts": failures}],
                   "best_config": best, "failure_counts": failures}
        return "STATE_JSON:\n" + json.dumps(payload)

    assert mock_proposal(state({"retrieval_miss": 9, "generation_miss": 1}))["config"]["top_k"] == 3
    assert mock_proposal(state({"retrieval_miss": 1, "generation_miss": 9}))["config"]["prompt_variant"] == "extractive"


# --- the loop ---


def test_tune_finds_the_better_config_and_scores_baseline_and_best_on_test(store):
    questions = make_questions(60)
    store.save_questions(questions)
    result = tune(store, questions)

    assert result.baseline.config == {"quality": 0, "flavor": "a"}
    assert result.best.config["quality"] == 2
    assert result.best.dev.score == 100.0
    assert result.baseline_test.split == "test" and result.best_test.split == "test"
    assert result.best_test.score == 100.0 and result.baseline_test.score == 0.0
    assert not result.overfit_warning
    assert result.stopped_reason == "search_space_exhausted"
    assert len(result.trials) == SPACE.size


def test_tune_respects_max_trials(store):
    questions = make_questions(30)
    store.save_questions(questions)
    result = tune(store, questions, max_trials=2)
    assert len(result.trials) == 2 and result.stopped_reason == "max_trials"


def test_overfitting_is_flagged_when_dev_beats_test(store):
    questions = make_questions(60)
    store.save_questions(questions)
    result = tune(store, questions, overfit=True)
    assert result.best.dev.score == 100.0
    assert result.best_test.score < 100.0 - 10
    assert result.overfit_warning
    assert result.overfit_gap > 10


def test_overfit_threshold_is_configurable(store):
    questions = make_questions(60)
    store.save_questions(questions)
    assert not tune(store, questions, overfit=True, overfit_threshold=1000).overfit_warning


def test_budget_too_small_for_the_final_test_evaluation_is_refused(store):
    questions = make_questions(30)
    store.save_questions(questions)
    with pytest.raises(TuneError, match="too small"):
        tune(store, questions, max_llm_calls=5)


def test_budget_stops_the_search_but_still_runs_the_final_test_evaluation(store):
    questions = make_questions(30)
    store.save_questions(questions)
    n_dev = len(store.dev_questions())
    n_test = len(questions) - n_dev
    result = tune(store, questions, max_llm_calls=2 * n_test + n_dev + n_dev // 2)
    assert result.stopped_reason == "max_llm_calls"
    assert len(result.trials) == 1
    assert result.best_test is not None
    assert result.llm_calls <= 2 * n_test + n_dev + n_dev // 2


def test_trials_and_result_are_saved_in_the_store(store):
    questions = make_questions(20)
    store.save_questions(questions)
    result = tune(store, questions, max_trials=3)
    assert [t.index for t in store.load_tune_trials(result.tune_id)] == [0, 1, 2]
    assert store.load_tune_result(result.tune_id).best_index == result.best_index


# --- the locked test set, checked while tune actually runs ---


def test_tune_only_reads_test_data_inside_the_locked_final_evaluation(store, monkeypatch):
    questions = make_questions(60)
    store.save_questions(questions)
    test_texts = {q.question for q in questions if q.split == Split.TEST}
    test_answers = {q.gold_answer for q in questions if q.split == Split.TEST}

    reads = []
    original = Store.load_questions

    def recording_load(self, split=None, statuses=None, **kwargs):
        reads.append((split, lock.is_test_split_unlocked()))
        return original(self, split=split, **({"statuses": statuses} if statuses is not None else {}))

    monkeypatch.setattr(Store, "load_questions", recording_load)

    final_calls = []
    original_final = tune_loop.evaluate_test_split

    def recording_final(*args, **kwargs):
        final_calls.append(kwargs["config"])
        return original_final(*args, **kwargs)

    monkeypatch.setattr(tune_loop, "evaluate_test_split", recording_final)

    proposer = MockLLM("mock-proposer")
    prompts = []
    original_complete = proposer.complete
    proposer.complete = lambda system, user: (prompts.append(user), original_complete(system, user))[1]

    run_tune(
        store, quality_factory(questions), "sha", "fake.py", SPACE, MockLLM("mock-judge"), "index-1",
        strategy="llm", proposer_client=proposer, max_trials=4,
    )

    test_reads = [r for r in reads if r[0] in (Split.TEST, None)]
    assert test_reads and all(unlocked for _, unlocked in test_reads)
    assert len(final_calls) == 2  # baseline and best, once each
    assert prompts, "the llm strategy should have been consulted"
    for prompt in prompts:
        assert not any(text in prompt for text in test_texts)
        assert not any(f'"{answer}"' in prompt for answer in test_answers)


# --- CLI ---


def test_cli_tune_on_the_demo_writes_a_best_config(tmp_path):
    runner = CliRunner()
    store_path = str(tmp_path / "store.db")
    corpus = str(DEMO_DIR / "corpus.jsonl")
    assert runner.invoke(app, ["generate", "--corpus", corpus, "--store", store_path, "--llm", "mock"]).exit_code == 0

    out = tmp_path / "best.yaml"
    result = runner.invoke(
        app,
        [
            "tune", "--adapter", str(DEMO_DIR / "toy_adapter.py"), "--search-space", str(DEMO_DIR / "search_space.yaml"),
            "--corpus", corpus, "--store", store_path, "--llm", "mock", "--strategy", "llm", "--max-trials", "4",
            "--out", str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Final scores" in result.output and "baseline" in result.output
    best = yaml_io.load(out)
    assert set(best) == {"chunk_size", "chunk_overlap", "top_k", "hybrid_weight", "reranker", "prompt_variant"}

    rerun = runner.invoke(
        app, ["evaluate", "--adapter", str(DEMO_DIR / "toy_adapter.py"), "--config", str(out), "--store", store_path, "--llm", "mock"]
    )
    assert rerun.exit_code == 0 and "mode cached" in rerun.output


def test_cli_tune_rejects_a_bad_search_space(tmp_path):
    space = tmp_path / "space.yaml"
    space.write_text("knobs:\n  top_k: [1, 3]\nbaseline:\n  top_k: 7\n")
    result = CliRunner().invoke(
        app, ["tune", "--adapter", str(DEMO_DIR / "toy_adapter.py"), "--search-space", str(space), "--store", str(tmp_path / "s.db"), "--llm", "mock"]
    )
    assert result.exit_code != 0 and "outside the search space" in result.output
    assert "Traceback" not in result.output

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragaudit import yaml_io
from ragaudit.adapters.base import AdapterLoadError, load_adapter_factory
from ragaudit.cli import app
from ragaudit.models.adapter import RAGResponse
from ragaudit.models.evaluation import FailureClass
from ragaudit.models.question import Question, Split
from ragaudit.models.scoring import CorrectnessResult, Verdict
from ragaudit.runner.evaluate import (
    classify,
    compute_config_hash,
    evaluate_question,
    run_evaluation,
    score_retrieval_at_k,
)
from ragaudit.runner.locked import evaluate_test_split
from ragaudit.storage.lock import LockedTestSetError
from ragaudit.storage.store import Store

DEMO_DIR = Path(__file__).resolve().parent.parent / "demo" / "v2"


def make_question(i: int, chunk_ids=None, split: Split | None = None) -> Question:
    """A question on chunk c{i}; pass split to search for the first i >= given i in that split."""
    while True:
        q = Question.create(
            chunk_ids=chunk_ids or [f"c{i}"],
            chunk_hashes=[f"h{i}" for _ in (chunk_ids or [0])],
            fact=f"fact {i}",
            question=f"question {i}?",
            gold_answer=f"gold {i}",
            generator_model="gen",
        )
        if split is None or q.split == split:
            return q
        i += 1000


def questions_in(split: Split, n: int, offset: int = 0) -> list[Question]:
    found, i = [], offset
    while len(found) < n:
        q = make_question(i)
        if q.split == split:
            found.append(q)
        i += 1
    return found


class FakeAdapter:
    """Retrieves the gold chunk and answers correctly for questions whose number is in `good`."""

    def __init__(self, good: set[int] | None = None, retrieve: bool = True):
        self.good = good
        self.retrieve = retrieve
        self.calls = 0

    def query(self, question: str) -> RAGResponse:
        self.calls += 1
        n = int(question.split()[1].rstrip("?"))
        correct = self.good is None or n in self.good
        return RAGResponse(
            answer=f"gold {n}" if correct else "no idea",
            retrieved_chunk_ids=[f"c{n}"] if self.retrieve else ["elsewhere"],
        )


def exact_judge(question: str, expected: str, candidate: str) -> CorrectnessResult:
    return CorrectnessResult(is_correct=expected == candidate, judge_model="exact-judge")


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "store.db") as s:
        yield s


def evaluate(store, questions, adapter, index="v1", config_hash="cfg", **kwargs):
    return run_evaluation(
        store, questions, questions[0].split, adapter, exact_judge,
        config={}, config_hash=config_hash, index_version=index, **kwargs,
    )


# --- scoring ---


def test_retrieval_metrics_for_single_and_multi_hop():
    assert score_retrieval_at_k(["a"], ["x", "a", "y"], k=5) == (1.0, 0.5, True)
    assert score_retrieval_at_k(["a"], ["x", "y", "a"], k=2) == (0.0, 0.0, False)
    assert score_retrieval_at_k(["a", "b"], ["b", "x"], k=5) == (0.5, 1.0, False)
    assert score_retrieval_at_k(["a", "b"], ["b", "a"], k=5) == (1.0, 1.0, True)


def test_failure_classes():
    assert classify(retrieval_hit=True, is_correct=True) == FailureClass.PASS
    assert classify(retrieval_hit=True, is_correct=False) == FailureClass.GENERATION_MISS
    assert classify(retrieval_hit=False, is_correct=False) == FailureClass.RETRIEVAL_MISS
    # A lucky pass is not a pass: it can't inflate the score.
    assert classify(retrieval_hit=False, is_correct=True) == FailureClass.RETRIEVAL_MISS


def test_correct_answer_without_retrieval_is_a_lucky_pass_retrieval_miss():
    result = evaluate_question(make_question(1), FakeAdapter(retrieve=False), exact_judge, k=5)
    assert result.failure_class == FailureClass.RETRIEVAL_MISS
    assert result.verdict == Verdict.LUCKY_PASS


def test_adapter_exception_is_a_scored_failure():
    class Broken:
        def query(self, question):
            raise RuntimeError("boom")

    result = evaluate_question(make_question(1), Broken(), exact_judge, k=5)
    assert result.failure_class == FailureClass.RETRIEVAL_MISS
    assert result.adapter_error == "boom"


def test_results_never_contain_the_gold_answer():
    result = evaluate_question(make_question(1), FakeAdapter(good=set()), exact_judge, k=5)
    assert "gold 1" not in result.model_dump_json()


def test_summary_score_is_pass_rate_in_points(store):
    questions = questions_in(Split.DEV, 4)
    good = {int(q.question.split()[1].rstrip("?")) for q in questions[:3]}
    summary, results = evaluate(store, questions, FakeAdapter(good=good))
    assert summary.score == 75.0
    assert summary.failure_counts == {"pass": 3, "retrieval_miss": 0, "generation_miss": 1}
    assert summary.mode == "full" and summary.n_run == 4


# --- cache ---


def test_same_config_and_index_reuses_cached_results(store):
    questions = questions_in(Split.DEV, 5)
    evaluate(store, questions, FakeAdapter())
    adapter = FakeAdapter()
    summary, _ = evaluate(store, questions, adapter)
    assert adapter.calls == 0
    assert summary.mode == "cached" and summary.n_cached == 5


def test_new_questions_run_even_when_index_is_unchanged(store):
    questions = questions_in(Split.DEV, 6)
    evaluate(store, questions[:4], FakeAdapter())
    adapter = FakeAdapter()
    summary, _ = evaluate(store, questions, adapter)
    assert adapter.calls == 2 and summary.n_cached == 4


def test_config_change_runs_everything(store):
    questions = questions_in(Split.DEV, 5)
    evaluate(store, questions, FakeAdapter())
    adapter = FakeAdapter()
    summary, _ = evaluate(store, questions, adapter, config_hash="other-config", index="v2", regression_sample=1)
    assert summary.mode == "full" and adapter.calls == 5


# --- regression sampling ---


def test_index_change_runs_new_questions_plus_a_seeded_sample_of_old(store):
    old, new = questions_in(Split.DEV, 20), questions_in(Split.DEV, 3, offset=500)
    evaluate(store, old, FakeAdapter(), index="v1")

    adapter = FakeAdapter()
    summary, results = evaluate(store, old + new, adapter, index="v2", regression_sample=5, seed=7)
    assert summary.mode == "regression"
    assert summary.n_new == 3 and summary.n_sampled_old == 5
    assert adapter.calls == 8
    assert {q.question_id for q in new} <= {r.question_id for r in results}


def test_regression_sample_is_deterministic_for_a_seed(tmp_path):
    old, new = questions_in(Split.DEV, 20), questions_in(Split.DEV, 2, offset=500)
    picked = []
    for name in ("a", "b"):
        with Store(tmp_path / f"{name}.db") as s:
            evaluate(s, old, FakeAdapter(), index="v1")
            _, results = evaluate(s, old + new, FakeAdapter(), index="v2", regression_sample=5, seed=3)
            picked.append(sorted(r.question_id for r in results))
    assert picked[0] == picked[1]


def test_full_flag_overrides_sampling(store):
    old = questions_in(Split.DEV, 10)
    evaluate(store, old, FakeAdapter(), index="v1")
    summary, _ = evaluate(store, old, FakeAdapter(), index="v2", regression_sample=2, full=True)
    assert summary.n_questions == 10


# --- the locked test split ---


def test_run_evaluation_refuses_mixed_splits(store):
    with pytest.raises(ValueError):
        run_evaluation(
            store, questions_in(Split.DEV, 1) + questions_in(Split.TEST, 1), Split.DEV, FakeAdapter(), exact_judge,
            config={}, config_hash="c", index_version="v",
        )


def test_test_split_evaluation_returns_aggregates_and_locks_its_results(store):
    test_qs = questions_in(Split.TEST, 3)
    store.save_questions(test_qs + questions_in(Split.DEV, 3))
    summary = evaluate_test_split(store, FakeAdapter(), exact_judge, config={}, config_hash="c", index_version="v")
    assert summary.split == "test" and summary.n_questions == 3 and summary.score == 100.0
    with pytest.raises(LockedTestSetError):
        store.load_result(test_qs[0].question_id, "c", "v")


# --- config hash and adapters ---


def test_config_hash_tracks_everything_that_changes_results():
    base = compute_config_hash({"top_k": 3}, "adapter-sha", "judge", 5)
    assert base == compute_config_hash({"top_k": 3}, "adapter-sha", "judge", 5)
    assert base != compute_config_hash({"top_k": 4}, "adapter-sha", "judge", 5)
    assert base != compute_config_hash({"top_k": 3}, "other-sha", "judge", 5)
    assert base != compute_config_hash({"top_k": 3}, "adapter-sha", "other-judge", 5)
    assert base != compute_config_hash({"top_k": 3}, "adapter-sha", "judge", 3)


def test_adapter_factory_supports_make_adapter_configure_and_plain_v1(tmp_path):
    make = tmp_path / "make.py"
    make.write_text(
        "class A:\n    def __init__(self, c): self.c = c\n    def query(self, q): return q\n"
        "def make_adapter(config): return A(config)\n"
    )
    assert load_adapter_factory(str(make))({"top_k": 2}).c == {"top_k": 2}

    configurable = tmp_path / "conf.py"
    configurable.write_text(
        "class A:\n    c = None\n    def configure(self, c): self.c = c\n    def query(self, q): return q\nadapter = A()\n"
    )
    assert load_adapter_factory(str(configurable))({"x": 1}).c == {"x": 1}

    plain = tmp_path / "plain.py"
    plain.write_text("def query(question):\n    return question\n")
    factory = load_adapter_factory(str(plain))
    assert factory({}).query("hi") == "hi"
    with pytest.raises(AdapterLoadError, match="doesn't accept a config"):
        factory({"top_k": 1})


# --- YAML ---


def test_yaml_subset_round_trip(tmp_path):
    data = {
        "knobs": {"chunk_size": [20, 40], "reranker": ["none", "overlap"], "hybrid_weight": {"min": 0.0, "max": 1.0, "step": 0.5}},
        "baseline": {"top_k": 1, "prompt_variant": "lead", "flag": True, "missing": None, "label": "a b: c"},
    }
    path = yaml_io.dump(data, tmp_path / "x.yaml", header="generated")
    assert yaml_io.load(path) == data


def test_yaml_block_lists_and_comments():
    text = "# comment\nknobs:\n  top_k:\n    - 1\n    - 3  # inline comment\n  name: 'quoted # not a comment'\n"
    assert yaml_io.loads(text) == {"knobs": {"top_k": [1, 3], "name": "quoted # not a comment"}}


# --- CLI ---


def test_cli_evaluate_dev_then_cached_then_test(tmp_path):
    runner = CliRunner()
    store_path = str(tmp_path / "store.db")
    corpus = str(DEMO_DIR / "corpus.jsonl")
    adapter = str(DEMO_DIR / "toy_adapter.py")
    assert runner.invoke(app, ["generate", "--corpus", corpus, "--store", store_path, "--llm", "mock"]).exit_code == 0

    first = runner.invoke(app, ["evaluate", "--adapter", adapter, "--corpus", corpus, "--store", store_path, "--llm", "mock"])
    assert first.exit_code == 0, first.output
    assert "mode full" in first.output and "Score (pass rate)" in first.output

    second = runner.invoke(app, ["evaluate", "--adapter", adapter, "--store", store_path, "--llm", "mock"])
    assert "mode cached" in second.output and " 0 run." in second.output

    config = tmp_path / "cfg.yaml"
    config.write_text("top_k: 1\nprompt_variant: lead\n")
    test = runner.invoke(
        app, ["evaluate", "--adapter", adapter, "--store", store_path, "--llm", "mock", "--split", "test", "--config", str(config)]
    )
    assert test.exit_code == 0, test.output
    assert "active test question(s)" in test.output


def test_cli_evaluate_without_questions_fails_cleanly(tmp_path):
    result = CliRunner().invoke(
        app,
        ["evaluate", "--adapter", str(DEMO_DIR / "toy_adapter.py"), "--corpus", str(DEMO_DIR / "corpus.jsonl"),
         "--store", str(tmp_path / "s.db"), "--llm", "mock"],
    )
    assert result.exit_code != 0
    assert "no active dev questions" in result.output
    assert "Traceback" not in result.output

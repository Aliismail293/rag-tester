"""--match-by text: retrieval hits from the contexts the adapter returns, not from chunk ids."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragaudit.cli import app
from ragaudit.runner.evaluate import compute_config_hash, score_retrieval_by_text_at_k
from ragaudit.scoring.retrieval import score_retrieval_by_text
from ragaudit.scoring.text_match import first_matching_context, text_match_score, validate_match_by
from ragaudit.storage import db as db_module
from ragaudit.storage import repository

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo" / "v2"

GOLD = (
    "The Riverside Line opened in 1991 and was resignalled in 2023. It runs from Millbrook Exchange "
    "to Ostley Quarry Gate and carries about forty thousand riders on a weekday."
)


# --- scoring ---


def test_identical_text_scores_one():
    assert text_match_score(GOLD, GOLD) == 1.0


def test_gold_inside_a_bigger_context_matches():
    context = "Some unrelated preamble about the authority's history. " + GOLD + " Then a paragraph about buses."
    assert text_match_score(GOLD, context) == 1.0


def test_context_that_is_a_slice_of_the_gold_chunk_matches():
    # The pipeline chunks smaller than ragaudit: its context is half of the gold chunk.
    half = GOLD[: len(GOLD) // 2]
    assert text_match_score(GOLD, half) >= 0.9


def test_same_topic_different_passage_does_not_match():
    other = (
        "The Harbour Line opened in 2004 and serves the docks. Riders on the Harbour Line can change at "
        "Millbrook Exchange for regional trains and the airport bus."
    )
    assert text_match_score(GOLD, other) < 0.5


def test_a_few_shared_words_cannot_match_everything():
    assert text_match_score(GOLD, "The Riverside Line") < 0.5
    assert text_match_score(GOLD, "") == 0.0


def test_matching_ignores_case_punctuation_and_whitespace():
    noisy = GOLD.upper().replace(" ", "  \n").replace(".", " ;")
    assert text_match_score(GOLD, noisy) == 1.0


def test_first_matching_context_gives_rank_and_best_score():
    rank, best = first_matching_context(GOLD, ["unrelated text about buses and trams here", GOLD, GOLD], 0.5)
    assert rank == 1 and best == 1.0
    assert first_matching_context(GOLD, [], 0.5) == (None, 0.0)


def test_threshold_decides_the_hit():
    partial = GOLD[:90] + " Buses leave the central station every ten minutes and the depot closes at midnight on Sundays."
    score = text_match_score(GOLD, partial)
    assert first_matching_context(GOLD, [partial], score)[0] == 0
    assert first_matching_context(GOLD, [partial], score + 0.01)[0] is None


def test_validate_match_by():
    assert validate_match_by("id") == "id" and validate_match_by("text") == "text"
    with pytest.raises(ValueError, match="id or text"):
        validate_match_by("fuzzy")


def test_v1_text_scorer_records_mode_and_score():
    result = score_retrieval_by_text("c1", GOLD, ["their-own-id-7"], ["noise words only here today", GOLD], 0.5)
    assert result.hit and result.rank == 1 and result.match_by == "text" and result.match_score == 1.0
    miss = score_retrieval_by_text("c1", GOLD, [], [], 0.5)
    assert not miss.hit and miss.rank is None and miss.match_score == 0.0


def test_v2_text_scoring_multi_hop_needs_every_gold_chunk():
    second = "Fenwick Depot, on the eastern edge of Millbrook, is the largest depot and houses ninety trams."
    recall, rr, hit = score_retrieval_by_text_at_k([GOLD, second], ["filler text that matches nothing at all", GOLD], k=5)
    assert recall == 0.5 and rr == 0.5 and not hit
    recall, rr, hit = score_retrieval_by_text_at_k([GOLD, second], [second, GOLD], k=5)
    assert recall == 1.0 and rr == 1.0 and hit
    assert score_retrieval_by_text_at_k([GOLD], ["x", "y", GOLD], k=2) == (0.0, 0.0, False)  # outside top k


def test_id_mode_config_hash_is_unchanged_and_text_mode_differs():
    base = compute_config_hash({"top_k": 3}, "sha", "judge", 5)
    assert base == compute_config_hash({"top_k": 3}, "sha", "judge", 5, None)
    text = compute_config_hash({"top_k": 3}, "sha", "judge", 5, 0.5)
    assert text != base and text != compute_config_hash({"top_k": 3}, "sha", "judge", 5, 0.6)


# --- v1 `run` CLI ---


class StubClient:
    def __init__(self, *args, **kwargs):
        self.model = "stub-model"

    def complete(self, system: str, user: str) -> str:
        if "generating a test set" in system:
            return '{"questions": [{"question": "What is in this document?", "answer": "some fact"}]}'
        if "judging whether a candidate answer" in system:
            return '{"is_correct": true, "score": 1.0, "rationale": "matches"}'
        if "judging whether an answer is faithfully grounded" in system:
            return '{"is_grounded": true, "score": 1.0, "rationale": "grounded"}'
        return "a plain no-context guess"


@pytest.fixture
def prepared_db(tmp_path, monkeypatch):
    """A v1 db with one ingested doc and one generated test case."""
    monkeypatch.setattr("ragaudit.cli.LLMClient", StubClient)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "doc1.txt").write_text(GOLD)
    db = tmp_path / "ragaudit.db"
    runner = CliRunner()
    assert runner.invoke(app, ["ingest", str(docs), "--db", str(db)]).exit_code == 0
    assert runner.invoke(app, ["generate", "--db", str(db)]).exit_code == 0
    return db


def own_index_adapter(tmp_path, contexts: list[str]) -> Path:
    """An adapter whose pipeline chunks the corpus itself: its ids mean nothing to ragaudit."""
    path = tmp_path / "own_index_adapter.py"
    path.write_text(
        "from ragaudit.models.adapter import RAGResponse\n\n"
        "def query(question):\n"
        f"    return RAGResponse(answer='some fact', retrieved_chunk_ids=['my-chunk-1'], contexts={contexts!r})\n"
    )
    return path


def latest_run(db: Path):
    conn = db_module.get_connection(db)
    try:
        return repository.load_run(conn, repository.get_latest_run_id(conn))
    finally:
        conn.close()


def run_cli(db: Path, adapter: Path, *extra: str):
    return CliRunner().invoke(app, ["run", "--adapter", str(adapter), "--db", str(db), "--no-ablation", *extra])


def test_run_defaults_to_id_matching(prepared_db, tmp_path):
    adapter = own_index_adapter(tmp_path, ["Intro. " + GOLD])
    result = run_cli(prepared_db, adapter)
    assert result.exit_code == 0, result.output
    run = latest_run(prepared_db)
    assert run.summary.retrieval_hit_rate == 0.0
    assert all(case.retrieval.match_by == "id" for case in run.scored_cases)


def test_run_match_by_text_hits_without_reindexing(prepared_db, tmp_path):
    adapter = own_index_adapter(tmp_path, ["Intro. " + GOLD[:120], GOLD[100:]])
    result = run_cli(prepared_db, adapter, "--match-by", "text")
    assert result.exit_code == 0, result.output
    assert "Matching retrieval by text" in result.output
    run = latest_run(prepared_db)
    assert run.summary.retrieval_hit_rate == 1.0
    case = run.scored_cases[0]
    assert case.retrieval.match_by == "text" and case.retrieval.rank == 0 and case.retrieval.match_score >= 0.5


def test_run_match_by_text_misses_an_unrelated_context(prepared_db, tmp_path):
    adapter = own_index_adapter(tmp_path, ["Buses run every ten minutes on weekdays from the central station."])
    assert run_cli(prepared_db, adapter, "--match-by", "text").exit_code == 0
    assert latest_run(prepared_db).summary.retrieval_hit_rate == 0.0


def test_run_match_by_text_threshold_is_configurable(prepared_db, tmp_path):
    # Half the context is the gold passage, half is something else: a score of about 0.5.
    partial = GOLD[:90] + " Buses leave the central station every ten minutes and the depot closes at midnight on Sundays."
    assert 0.3 < text_match_score(GOLD, partial) < 0.7
    adapter = own_index_adapter(tmp_path, [partial])
    assert run_cli(prepared_db, adapter, "--match-by", "text", "--match-threshold", "0.2").exit_code == 0
    assert latest_run(prepared_db).summary.retrieval_hit_rate == 1.0
    assert run_cli(prepared_db, adapter, "--match-by", "text", "--match-threshold", "0.99").exit_code == 0
    assert latest_run(prepared_db).summary.retrieval_hit_rate == 0.0


def test_run_match_by_text_warns_when_adapter_returns_no_contexts(prepared_db, tmp_path):
    result = run_cli(prepared_db, own_index_adapter(tmp_path, []), "--match-by", "text")
    assert result.exit_code == 0
    assert "returned no `contexts`" in result.output


def test_run_rejects_unknown_match_mode(prepared_db, tmp_path):
    result = run_cli(prepared_db, own_index_adapter(tmp_path, [GOLD]), "--match-by", "fuzzy")
    assert result.exit_code != 0 and "id or text" in result.output and "Traceback" not in result.output


# --- v2 `evaluate` CLI ---


def test_evaluate_match_by_text_needs_corpus_and_caches_separately(tmp_path):
    runner = CliRunner()
    store = str(tmp_path / "store.db")
    corpus = str(DEMO_DIR / "corpus.jsonl")
    adapter = str(DEMO_DIR / "toy_adapter.py")
    base = ["evaluate", "--adapter", adapter, "--store", store, "--llm", "mock"]
    assert runner.invoke(app, ["generate", "--corpus", corpus, "--store", store, "--llm", "mock"]).exit_code == 0

    no_corpus = runner.invoke(app, [*base, "--match-by", "text"])
    assert no_corpus.exit_code != 0 and "--corpus" in no_corpus.output

    by_id = runner.invoke(app, [*base, "--corpus", corpus])
    assert by_id.exit_code == 0, by_id.output
    by_text = runner.invoke(app, [*base, "--corpus", corpus, "--match-by", "text"])
    assert by_text.exit_code == 0, by_text.output
    assert "Matching retrieval by text" in by_text.output
    assert "mode full" in by_text.output  # text mode is a different config: no id-mode cache reuse

    again = runner.invoke(app, [*base, "--corpus", corpus, "--match-by", "text"])
    assert "mode cached" in again.output

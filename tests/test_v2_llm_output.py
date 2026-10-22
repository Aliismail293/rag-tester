"""Regressions from a real-LLM smoke test (qwen2.5:7b generator, llama3.1:8b verifier via Ollama).

Each raw string below is, or is trimmed from, an actual model reply that the
pipeline previously lost or let through.
"""

import json

import pytest

from ragaudit.generate import prompts
from ragaudit.generate.pipeline import (
    build_checked_question,
    deword_question,
    is_usable_fact,
    parse_facts_response,
    question_problem,
    write_question,
)
from ragaudit.llm.json_output import parse_json_object
from ragaudit.models.document import Chunk

PASSAGE = (
    "Line 1, known as the Riverside Line, is HVTA's oldest tram line and opened in 1991. The Riverside Line "
    "was resignalled in 2023. HVTA publishes its timetables in March and September."
)


class Reply:
    """Answers every call with the next scripted reply."""

    def __init__(self, model: str, *replies: str):
        self.model = model
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.replies.pop(0)


def chunk() -> Chunk:
    return Chunk(chunk_id="line1-01", doc_id="line1", position=0, text=PASSAGE)


# --- JSON extraction ---


def test_facts_reply_missing_its_final_brace_is_recovered():
    # qwen2.5:7b dropped the closing "}" on 2 of 10 chunks, losing every fact in them.
    raw = (
        '{"facts": [{"fact": "The Harrowgate Valley Transit Authority was established in 1987.", "answer": "1987"}, '
        '{"fact": "The head office of the Harrowgate Valley Transit Authority is on Corran Street in Millbrook.", '
        '"answer": "Corran Street in Millbrook"}]'
    )
    facts = parse_facts_response(raw)
    assert facts is not None and [f["answer"] for f in facts] == ["1987", "Corran Street in Millbrook"]


@pytest.mark.parametrize(
    "raw",
    [
        '{"question": "What year?"}',
        'Sure! Here is the JSON:\n{"question": "What year?"}\nHope that helps.',
        'Here you go:\n```json\n{"question": "What year?"}\n```',
        '{"question": "What year?"',
    ],
)
def test_json_object_is_found_inside_common_wrappers(raw):
    assert parse_json_object(raw) == {"question": "What year?"}


def test_raw_newlines_inside_strings_are_parsed():
    assert parse_json_object('{"reason": "line one\nline two", "ok": true}') == {"reason": "line one\nline two", "ok": True}


@pytest.mark.parametrize("raw", ["", "no json here", '{"question": "unterminated', "[1, 2]", '{"a": 1]'])
def test_unrecoverable_replies_are_none(raw):
    assert parse_json_object(raw) is None


# --- fact sanity ---


@pytest.mark.parametrize(
    "fact, answer",
    [
        ("9", "9"),  # a bare value, not a fact
        ("Fenwick Depot", "Fenwick Depot"),
        ("Line 2 is", "11.2 kilometres long"),  # a fragment that doesn't contain its answer
        ("Line 2 connects Ostley Quarry Gate", "Millbrook Exchange"),
        ("November and February", "between November and February"),
    ],
)
def test_degenerate_facts_from_a_real_model_are_dropped(fact, answer):
    assert not is_usable_fact(fact, answer)
    assert parse_facts_response(json.dumps({"facts": [{"fact": fact, "answer": answer}]})) == []


@pytest.mark.parametrize(
    "fact, answer",
    [
        ("HVTA employs 1,240 people, of whom 610 are drivers.", "1,240 people"),
        ("Fenwick Depot, on the eastern edge of Millbrook, is HVTA's largest depot.", "Millbrook"),
        ("The heritage section is a 2.1-kilometre stretch through Ashby Old Town.", "2.1"),
        ("HVTA employs 1240 people", "1240"),
    ],
)
def test_real_facts_are_kept(fact, answer):
    assert is_usable_fact(fact, answer)


def test_numeric_answers_are_accepted_as_text():
    raw = '{"facts": [{"fact": "The Riverside Line opened in 1991.", "answer": 1991}, {"fact": "It is 14.6 km long.", "answer": 14.6}]}'
    assert parse_facts_response(raw) == [
        {"fact": "The Riverside Line opened in 1991.", "answer": "1991"},
        {"fact": "It is 14.6 km long.", "answer": "14.6"},
    ]


# --- question sanity ---


@pytest.mark.parametrize(
    "question, problem",
    [
        ("In which year did the Riverside Line undergo重新信号化改造？", "writing system"),
        ("What month does HVTA release its timetables according to the passage?", "refers to the passage"),
        ("In the given text, when did Line 1 open?", "refers to the passage"),
        ("From which depot is Line 1 managed by HVTA?\n\nOptions:\nA) Weirside Depot\nB) Fenwick Depot", "several lines"),
    ],
)
def test_bad_questions_from_a_real_model_are_caught(question, problem):
    assert problem in question_problem(question, PASSAGE)


@pytest.mark.parametrize(
    "question",
    ["In which year did the HVTA's Riverside Line become operational?", "What text messages does HVTA send riders?"],
)
def test_ordinary_questions_pass(question):
    assert question_problem(question, PASSAGE) is None


def test_a_rewrite_that_switches_language_falls_back_to_the_draft():
    draft = "When was the Riverside Line resignalled?"
    generator = Reply("gen", json.dumps({"question": "In which year did the Riverside Line undergo重新信号化改造？"}))
    assert deword_question(generator, draft, [chunk()]) is None

    generator = Reply(
        "gen",
        json.dumps({"question": "In which year did the Riverside Line undergo重新信号化改造？"}),
        json.dumps({"question": "In which year did the Riverside Line undergo重新信号化改造？"}),
    )
    verifier = Reply("ver", json.dumps({"reason": "ok", "supported": True, "answerable": True, "unambiguous": True}))
    question, _ = build_checked_question(generator, verifier, [chunk()], "The Riverside Line was resignalled in 2023.", "2023", draft, 0.9)
    assert question.question == draft


def test_a_draft_pointing_at_the_passage_yields_no_question():
    generator = Reply("gen", json.dumps({"question": "According to the passage, when did Line 1 open?"}))
    assert write_question(generator, "Line 1 opened in 1991.", "1991", [chunk()]) is None


# --- prompts ---


def test_verifier_prompt_asks_for_reasoning_first_and_names_multi_value_answers():
    prompt = prompts.VERIFY_SYSTEM_PROMPT
    assert prompt.index('"reason"') < prompt.index('"supported"')
    assert "more than one value" in prompt and "the passage" in prompt


def test_facts_prompt_asks_for_whole_lists_and_verbatim_answers():
    prompt = prompts.FACTS_SYSTEM_PROMPT
    assert "word for word" in prompt and "whole list" in prompt

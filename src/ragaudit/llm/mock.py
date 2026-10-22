"""A deterministic, offline stand-in for an LLM, used by tests and the offline v2 demo.

`MockLLM` answers every prompt ragaudit sends (fact extraction, question
writing, de-wording, verification, multi-hop, the correctness and
groundedness judges, tuning proposals) and the generic "Context: ...
Question: ..." answering prompt the demo adapter uses. Same input, same
output, no network. It works with sentence splitting, regexes and a small
synonym table, so its questions read like terse search-box queries rather
than polished prose.

It is deliberately imperfect, the way a real generator is, so the checks
downstream have something to catch:

- about one fact in ten gets a wrong number in its gold answer (the
  verifier should reject it as `unsupported`);
- its first de-wording attempt is a light synonym swap that usually
  still copies the passage (forcing the retry), and about one question in
  nine resists rewording entirely (`too_lexical`);
- it copies relative time phrases ("last quarter") from the passage
  instead of converting them (the relative-time validator should reject
  those).

It has no world knowledge: asked to answer without context, it says it
doesn't know.
"""

import hashlib
import json
import re

from ragaudit.generate import prompts
from ragaudit.generate.text import STOPWORDS, content_tokens, find_relative_time, tokenize

NO_CONTEXT_MARKER = "using only your own internal knowledge"
PROPOSER_MARKER = "You are tuning a RAG pipeline"
LEAD_ANSWER_MARKER = "first sentence"

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE_RE = re.compile(
    rf"\b(?:(?:{_MONTHS})\s+\d{{1,2}},\s*\d{{4}}|(?:{_MONTHS})\s+\d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b"
)
_TIME_RE = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?![\d:])")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_UNITS = (
    "percent|business days|calendar days|days?|hours?|minutes?|weeks?|months?|years?|seconds?|"
    "kilometres|kilometers|metres|meters|centimetres|km/h|kWh|km|kg|euros?|dollars?|"
    "people|passengers|stops|trams|buses|vehicles|technicians|controllers|drivers|employees|staff|"
    "members|units|seats|routes|lines|boardings|trips"
)
_NUMBER_RE = re.compile(
    rf"(?<![A-Za-z0-9.,-])(?:[$€£]\s?)?\d+(?:,\d{{3}})*(?:\.\d+)?(?:\s(?:million|billion))?(?:\s?%|\s(?:{_UNITS})\b)?(?![\d,.]\d)"
)
_NAME_RE = re.compile(r"\b[A-Z][a-zA-Z0-9&'-]*(?:\s+(?:of\s+|the\s+)?[A-Z0-9][a-zA-Z0-9&'-]*)*")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_WH_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "why", "how"}
_OPENERS = ("quick q: ", "remind me, ", "anyone know: ", "trying to recall: ")
_WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}

_SYNONYMS = {
    "established": "set up",
    "founded": "started",
    "opened": "started running",
    "employs": "has on staff",
    "employ": "have on staff",
    "governed": "overseen",
    "chair": "head",
    "held": "happen",
    "ridership": "passenger numbers",
    "reached": "hit",
    "carried": "moved",
    "busiest": "most crowded",
    "handles": "sees",
    "publishes": "puts out",
    "timetables": "schedules",
    "answers": "picks up",
    "property": "items",
    "oldest": "first",
    "runs": "goes",
    "run": "come",
    "leaves": "departs",
    "departs": "heads out",
    "journey": "ride",
    "trip": "ride",
    "takes": "lasts",
    "resignalled": "got new signals",
    "cost": "price",
    "costs": "is priced at",
    "closed": "shut",
    "maintained": "looked after",
    "floods": "goes underwater",
    "replacement": "backup",
    "standby": "reserve",
    "connects": "links",
    "long": "in length",
    "quarter": "three months",
    "performance": "punctuality",
    "caused": "due",
    "exclusively": "only",
    "limited": "capped",
    "newest": "most recent",
    "built": "constructed",
    "operate": "run",
    "section": "stretch",
    "uses": "relies on",
    "smallest": "tiniest",
    "trialling": "testing",
    "fare": "ticket",
    "valid": "good",
    "transfers": "changes",
    "pass": "ticket",
    "pay": "are charged",
    "travel": "ride",
    "penalty": "fine",
    "issued": "handed out",
    "accepted": "taken",
    "capped": "limited",
    "spending": "charges",
    "largest": "biggest",
    "space": "room",
    "houses": "contains",
    "stores": "parks",
    "extended": "enlarged",
    "towed": "hauled",
    "garage": "depot",
    "charging": "plug-in",
    "converted": "switched",
    "fleet": "vehicles",
    "consists": "is made up",
    "scheduled": "planned",
    "retirement": "withdrawal",
    "carries": "holds",
    "seated": "sitting",
    "battery": "cell pack",
    "range": "distance per charge",
    "ordered": "bought",
    "order": "purchase",
    "worth": "valued at",
    "receives": "gets",
    "inspection": "check",
    "inspections": "checks",
    "overhaul": "rebuild",
    "reprofiled": "reshaped",
    "carried out": "done",
    "technicians": "engineers",
    "inspected": "checked",
    "replaced": "swapped",
    "warranty": "guarantee",
    "access": "entry",
    "raised": "lifted",
    "assistance": "help",
    "book": "reserve",
    "ramp": "slope",
    "announcements": "callouts",
    "monitors": "watches",
    "staffed": "crewed",
    "controllers": "dispatchers",
    "recorded": "logged",
    "collisions": "crashes",
    "training": "instruction",
    "reassessment": "re-test",
    "report": "flag",
    "tested": "checked",
    "probation": "trial period",
    "entitled": "allowed",
    "leave": "holiday",
    "shifts": "duties",
    "break": "pause",
    "discount": "reduction",
    "claims": "requests",
    "submitted": "sent in",
    "approval": "sign-off",
    "paid": "reimbursed",
    "lengthen": "extend",
    "adding": "with",
    "construction": "building work",
    "budget": "funding",
    "expects": "forecasts",
    "redesign": "rebuild",
    "merge": "combine",
    "approved": "signed off",
    "replace": "swap out",
    "estimates": "reckons",
    "energy": "power",
    "every": "each",
    "must": "have to",
    "within": "inside",
    "maximum": "most",
    "minimum": "least",
    "annual": "yearly",
    "customer": "client",
    "employees": "staff",
}


def _h(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _field(user: str, label: str) -> str | None:
    match = re.search(rf"^{re.escape(label)}: (.*)$", user, re.MULTILINE)
    return match.group(1).strip() if match else None


def _passages(user: str) -> list[str]:
    numbered = re.split(r"^Passage \d+:\n", user, flags=re.MULTILINE)
    if len(numbered) > 1:
        return [p.strip() for p in numbered[1:]]
    single = user.split("Passage:\n", 1)
    return [single[1].strip()] if len(single) == 2 else []


def _sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    joined = " ".join(lines)
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(joined) if s.strip()]


def _normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def _contains(haystack: str, needle: str) -> bool:
    needle_norm = _normalize_text(needle)
    return bool(needle_norm) and f" {needle_norm} " in f" {_normalize_text(haystack)} "


def _is_name_number(sentence: str, start: int) -> bool:
    """A number right after a capitalized word ("Line 2", "Varda T-40") is part of a name."""
    before = sentence[:start].rstrip()
    previous_word = before.split()[-1] if before.split() else ""
    return bool(previous_word) and previous_word[0].isupper() and before == sentence[:start].rstrip()


def _answer_span(sentence: str) -> str | None:
    """A short answer span: a date, a time, a number (with unit), else the last non-initial name."""
    for regex in (_DATE_RE, _TIME_RE):
        match = regex.search(sentence)
        if match:
            return match.group(0)
    for match in _NUMBER_RE.finditer(sentence):
        if not _is_name_number(sentence, match.start()):
            return match.group(0).strip()
    names = [
        m.group(0)
        for m in _NAME_RE.finditer(sentence)
        if m.start() > 0 and not m.group(0)[0].isdigit() and not m.group(0).endswith("'s")
    ]
    return names[-1] if names else None


def _perturb_number(answer: str) -> str:
    """Bump the last digit run by one: the mock generator's deliberate hallucination."""
    last = list(re.finditer(r"\d+", answer))[-1]
    bumped = str(int(last.group(0)) + 1).zfill(len(last.group(0)))
    return answer[: last.start()] + bumped + answer[last.end() :]


def _extract_facts(passage: str, max_facts: int) -> list[dict[str, str]]:
    facts = []
    for sentence in _sentences(passage):
        if len(sentence.split()) < 6:
            continue
        answer = _answer_span(sentence)
        if answer is None or len(answer) >= len(sentence) * 0.5:
            continue
        fact = sentence
        if re.search(r"\d", answer) and _h(f"perturb:{sentence}") % 10 == 0:
            wrong = _perturb_number(answer)
            fact = sentence.replace(answer, wrong, 1)
            answer = wrong
        facts.append({"fact": fact, "answer": answer})
        if len(facts) >= max_facts:
            break
    return facts


def _write_question(fact: str, answer: str) -> str:
    """The mock's first draft: the fact with its answer replaced by a wh-word. Copies the wording on purpose."""
    wh = "when" if _DATE_RE.fullmatch(answer) else "what"
    body = fact.replace(answer, wh, 1).rstrip(" .!")
    return body[0].upper() + body[1:] + "?"


def _swap(word: str) -> str:
    return _SYNONYMS.get(word.lower(), word)


def _mild_paraphrase(question: str) -> str:
    out = []
    for word in question.rstrip("?").split():
        bare = word.strip(",.;:")
        if bare.lower() in {"the", "a", "an"}:
            continue
        out.append(_swap(bare) if bare.lower() in _SYNONYMS else word)
    text = " ".join(out)
    return text[0].upper() + text[1:] + "?"


def _ask(fact: str, answer: str) -> str:
    """How a user would ask for this kind of answer."""
    lowered = answer.lower()
    before = fact[: fact.find(answer)].lower().split()
    if _TIME_RE.fullmatch(answer):
        return "what time"
    if _DATE_RE.fullmatch(answer):
        return "what date"
    if _YEAR_RE.fullmatch(answer):
        return "which year"
    if lowered in _WEEKDAYS:
        return "which day"
    if lowered[:1] in "$€£" or re.search(r"\b(euros?|dollars?)$", lowered):
        return "how much"
    if lowered.endswith(("%", "percent")):
        return "what percentage"
    if re.search(r"\b(minutes?|hours?|days?|weeks?|months?|years?|seconds?)$", lowered):
        return "how often" if before and before[-1] == "every" else "how long"
    if re.search(r"\b(kilometres|kilometers|metres|meters|centimetres|km)$", lowered):
        return "how long"
    unit = re.search(r"\d\s+([a-z]+)$", lowered)
    if unit:
        return f"how many {unit.group(1)}"
    if re.search(r"\d", answer):
        return "what number"
    return "what is the name"


def _is_real_name(name: str, at_start: bool) -> bool:
    """Drop capitalized sentence-initial common words ("Board", "Annual") that aren't names."""
    if not at_start:
        return True
    return len(name.split()) > 1 or sum(c.isupper() for c in name) > 1 or any(c.isdigit() for c in name)


def _subject(fact: str, answer: str) -> str:
    """The named entity the fact is about: the shortest real name that isn't the answer.

    Shortest, because a real user abbreviates ("HVTA", not the full
    authority name) and long names would dominate lexical overlap.
    """
    text = fact.replace(answer, " ")
    names = []
    for match in _NAME_RE.finditer(text):
        name = re.sub(r"^(The|A|An|Every|All|Each|In|This|Last|Its|It)\s+", "", match.group(0))
        if name in {"The", "A", "An", "Every", "All", "Each", "In", "This", "Last", "Its", "It"}:
            continue
        if _is_real_name(name, match.start() == 0 and name == match.group(0)) and not name.endswith("'s"):
            names.append(name)
    return min(names, key=lambda n: (len(n.split()), -len(n))) if names else ""


def _context_words(fact: str, answer: str, subject: str) -> list[str]:
    """Up to two content words just before the answer and one just after, synonym-swapped."""
    index = fact.find(answer)
    subject_tokens = content_tokens(subject)
    words_before = [w.strip(",.;:()") for w in fact[:index].split()]
    words_after = [w.strip(",.;:()") for w in fact[index + len(answer) :].split()]

    def keep(word: str) -> bool:
        return bool(word) and word.lower() not in STOPWORDS and not (content_tokens(word) & subject_tokens)

    before = [w for w in words_before if keep(w)][-3:]
    after = [w for w in words_after if keep(w)][:2]
    # Closest to the answer first: the last words before it, then the first after it.
    candidates = list(reversed(before)) + after
    reworded = [w for w in candidates if w.lower() in _SYNONYMS][:1]
    verbatim = [w for w in candidates if w.lower() not in _SYNONYMS][:1]
    chosen = [w for w in candidates if w in reworded + verbatim] or candidates[:1]
    return [_swap(w).lower() for w in chosen]


def _strong_rewrite(fact: str, answer: str) -> str:
    """A terse, half-remembered query: subject, what's being asked, a couple of swapped context words."""
    subject = _subject(fact, answer)
    if not subject:
        answer_tokens = content_tokens(answer)
        subject = " ".join([w for w in fact.split() if content_tokens(w) and not content_tokens(w) & answer_tokens][:2])
    ask = _ask(fact, answer)
    context = " ".join(_context_words(fact, answer, subject))
    opener = _OPENERS[_h(f"opener:{fact}") % len(_OPENERS)]
    question = " ".join(part for part in (f"{opener}{subject}".strip(), ask, context) if part) + "?"
    relative = find_relative_time(fact)
    if relative and not find_relative_time(question):
        question = f"{relative} {question}"
    return question[0].upper() + question[1:]


def _match_draft(draft: str, passages: list[str]) -> tuple[str, str] | None:
    """Recover (fact, answer) for a draft question written by _write_question from one of these passages."""
    for passage in passages:
        for sentence in _sentences(passage):
            answer = _answer_span(sentence)
            if answer and _write_question(sentence, answer) == draft:
                return sentence, answer
    return None


def _deword(user: str) -> str:
    draft = _field(user, "Question") or ""
    is_retry = _field(user, "Previous rewrite (still too close to the passage wording, change it more)") is not None
    if not is_retry or _h(f"stubborn:{draft}") % 9 == 0:
        return _mild_paraphrase(draft)
    matched = _match_draft(draft, _passages(user))
    if matched is None:
        return _mild_paraphrase(_mild_paraphrase(draft))
    return _strong_rewrite(*matched)


def _verify(fact: str, question: str, gold: str, passages: list[str]) -> dict:
    source = "\n".join(passages)
    gold_parts = [p.strip() for p in gold.split(";") if p.strip()]
    fact_tokens = content_tokens(fact)
    fact_support = len(fact_tokens & content_tokens(source)) / len(fact_tokens) if fact_tokens else 0.0
    supported = bool(gold_parts) and all(_contains(source, part) for part in gold_parts) and fact_support >= 0.8
    answer_leaked = any(_contains(question, part) for part in gold_parts)
    answerable = not answer_leaked and bool(content_tokens(question) & content_tokens(source))
    gold_tokens = set().union(*(content_tokens(p) for p in gold_parts)) if gold_parts else set()
    anchors = content_tokens(question) & (fact_tokens - gold_tokens)
    unambiguous = len(anchors) >= 2

    if not supported:
        reason = "the gold answer or fact is not stated in the passage"
    elif not answerable:
        reason = "the question gives away the answer" if answer_leaked else "the question is not answerable"
    elif not unambiguous:
        reason = "the question does not pin down which fact it asks about"
    else:
        reason = "ok"
    return {"supported": supported, "answerable": answerable, "unambiguous": unambiguous, "reason": reason}


def _multi_hop(passages: list[str]) -> dict | None:
    picked = []
    for passage in passages[:2]:
        facts = [f for f in _extract_facts(passage, 4) if _contains(passage, f["answer"])]
        if not facts:
            return None
        picked.append(facts[-1] if picked and facts[0]["answer"] == picked[0]["answer"] else facts[0])
    first, second = picked
    part_a = _strong_rewrite(first["fact"], first["answer"]).rstrip("?")
    part_b = _strong_rewrite(second["fact"], second["answer"]).rstrip("?")
    return {
        "fact": f"{first['fact']} {second['fact']}",
        "question": f"{part_a}, and {part_b[0].lower()}{part_b[1:]}?",
        "answer": f"{first['answer']}; {second['answer']}",
    }


def _judge_correctness(user: str) -> dict:
    expected = _field(user, "Expected answer") or ""
    candidate = _field(user, "Candidate answer") or ""
    parts = [p.strip() for p in expected.split(";") if p.strip()]
    candidate_tokens = set(tokenize(candidate))
    correct = bool(parts) and all((content_tokens(part) or set(tokenize(part))) <= candidate_tokens for part in parts)
    return {
        "is_correct": correct,
        "score": 1.0 if correct else 0.0,
        "rationale": "expected answer found in candidate" if correct else "expected answer missing from candidate",
    }


def _judge_groundedness(user: str) -> dict:
    contexts_part, _, answer_part = user.partition("\n\nAnswer:\n")
    answer = answer_part.split("\n\nIs the answer grounded", 1)[0]
    answer_tokens = content_tokens(answer)
    support = len(answer_tokens & content_tokens(contexts_part)) / len(answer_tokens) if answer_tokens else 0.0
    return {"is_grounded": support >= 0.8, "score": round(support, 3), "rationale": "token support check"}


def _answer_from_context(system: str, user: str) -> str:
    """Extractive answering: the context sentence sharing the most words with the question."""
    match = re.search(r"Context:\n(.*?)\n\nQuestion: (.*)", user, re.DOTALL)
    if not match:
        return "I don't know."
    context_block, question = match.group(1), match.group(2)
    contexts = [c for c in context_block.split("\n---\n") if c.strip() and c.strip() != "(no context retrieved)"]
    if not contexts:
        return "I don't know."
    if LEAD_ANSWER_MARKER in system.lower():
        sentences = _sentences(contexts[0])
        return sentences[0] if sentences else "I don't know."

    question_tokens = content_tokens(question)
    best, best_score = None, 0
    for context in contexts:
        for sentence in _sentences(context):
            score = len(question_tokens & content_tokens(sentence))
            if score > best_score:
                best, best_score = sentence, score
    return best or "I don't know."


class MockLLM:
    """Deterministic offline Completer. See the module docstring for its deliberate flaws."""

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self._dispatch(system, user)

    def _dispatch(self, system: str, user: str) -> str:
        # Imported lazily so importing the llm package never drags in scoring or generation.
        from ragaudit.generate import testset_generator
        from ragaudit.scoring import correctness, groundedness

        if system == prompts.FACTS_SYSTEM_PROMPT:
            max_facts = int(_field(user, "Maximum facts") or 3)
            passages = _passages(user)
            return json.dumps({"facts": _extract_facts(passages[0], max_facts) if passages else []})

        if system == prompts.QUESTION_SYSTEM_PROMPT:
            return json.dumps({"question": _write_question(_field(user, "Fact") or "", _field(user, "Answer") or "")})

        if system == prompts.DEWORD_SYSTEM_PROMPT:
            return json.dumps({"question": _deword(user)})

        if system == prompts.VERIFY_SYSTEM_PROMPT:
            return json.dumps(
                _verify(
                    _field(user, "Fact") or "",
                    _field(user, "Question") or "",
                    _field(user, "Gold answer") or "",
                    _passages(user),
                )
            )

        if system == prompts.MULTI_HOP_SYSTEM_PROMPT:
            result = _multi_hop(_passages(user))
            return json.dumps(result) if result else "no multi-hop question possible"

        if system == correctness.SYSTEM_PROMPT:
            return json.dumps(_judge_correctness(user))

        if system == groundedness.SYSTEM_PROMPT:
            return json.dumps(_judge_groundedness(user))

        if system.startswith(PROPOSER_MARKER):
            from ragaudit.tune import strategies

            return json.dumps(strategies.mock_proposal(user))

        if system == testset_generator.SYSTEM_PROMPT:
            facts = _extract_facts(user.split("\n\n", 1)[-1], 1)
            questions = [{"question": _strong_rewrite(f["fact"], f["answer"]), "answer": f["answer"]} for f in facts]
            return json.dumps({"questions": questions})

        if NO_CONTEXT_MARKER in system:
            return "I don't know."

        return _answer_from_context(system, user)

"""Prompts for v2 generation, and the user-prompt builders that pair with them.

User prompts use a fixed "Label: value" header followed by numbered
passages, so the offline MockLLM can parse them the same way every time.
"""

FACTS_SYSTEM_PROMPT = (
    "You extract atomic, verifiable facts from a passage, for building a retrieval test set. "
    "Each fact must be a single self-contained statement that is stated explicitly in the passage, "
    "is specific to it (not general knowledge), and has a short, unambiguous answer: a name, number, "
    "date, duration, place or short phrase taken from the passage. Write each fact so it makes sense "
    "on its own, naming its subject instead of using pronouns. Never invent or infer anything that is "
    "not written in the passage.\n"
    "\n"
    "Each fact is a full sentence, and its answer is copied word for word from that sentence. When the "
    "passage gives several values for the same thing (two months, a list of names), keep them together in "
    "one fact whose answer is the whole list, so every question has exactly one correct answer.\n"
    "\n"
    'Example: {"fact": "The Zentrix vault in Pellow opens every Thursday.", "answer": "every Thursday"}\n'
    "\n"
    "Respond with strict JSON only, no prose, no markdown fences, in the form: "
    '{"facts": [{"fact": str, "answer": str}, ...]}'
)

QUESTION_SYSTEM_PROMPT = (
    "You write one question that tests a single fact, for a retrieval test set. The question must be "
    "answerable from the fact, have exactly one correct answer (the given answer), and name its subject "
    "explicitly so it is unambiguous without the passage. Never refer to \"the passage\", \"the document\" "
    "or \"the text\", and never include the answer in the question.\n"
    "\n"
    "Never use relative time words such as yesterday, today, tomorrow, now, currently, recently, lately, "
    "latest, last week, this year or \"two years ago\". If time matters, use absolute dates or years; "
    "the passage's capture date is given when known.\n"
    "\n"
    'Respond with strict JSON only, no prose, no markdown fences, in the form: {"question": str}'
)

DEWORD_SYSTEM_PROMPT = (
    "You rewrite a test question so it does not reuse the distinctive wording of the source passage. "
    "Phrase it the way a real user who half-remembers the topic would ask: use their own everyday words, "
    "synonyms and a different sentence structure. Keep the exact meaning, keep names needed to make it "
    "unambiguous, keep exactly one correct answer, and do not include the answer. Never introduce "
    "relative time words (now, currently, recently, latest, last year, ...).\n"
    "\n"
    'Respond with strict JSON only, no prose, no markdown fences, in the form: {"question": str}'
)

VERIFY_SYSTEM_PROMPT = (
    "You are an independent verifier for a retrieval test set. Most generated questions have at least one "
    "problem; your job is to find it. Given source passage(s), a fact, a question and a gold answer, check "
    "three things strictly:\n"
    "1. supported: the fact AND the gold answer are explicitly supported by the passage(s).\n"
    "2. answerable: the question can be answered from the passage(s), and the gold answer is the complete "
    "answer to it.\n"
    "3. unambiguous: the question has exactly one correct answer and names its subject clearly enough "
    "that a reader without the passage knows what is being asked. It is NOT unambiguous if the passage(s) "
    "give more than one value that fits (for example two months, two dates or a list) and the gold answer "
    "names only one, if it points at \"the passage\" or \"the text\" instead of naming its subject, or if "
    "it is not written entirely in the passage's language.\n"
    "\n"
    "First explain your checks in \"reason\", then give the verdicts. Respond with strict JSON only, no "
    "prose, no markdown fences, in the form: "
    '{"reason": str, "supported": bool, "answerable": bool, "unambiguous": bool}'
)

MULTI_HOP_SYSTEM_PROMPT = (
    "You write one multi-hop test question that can only be answered by combining information from BOTH "
    "passages: neither passage alone is enough. Also state the combined fact it tests and the short gold "
    "answer. Name subjects explicitly, never refer to \"the passage\", never include the answer in the "
    "question, and never use relative time words (now, currently, recently, latest, last year, ...).\n"
    "\n"
    "Respond with strict JSON only, no prose, no markdown fences, in the form: "
    '{"fact": str, "question": str, "answer": str}'
)


def _one_line(value: str) -> str:
    return " ".join(value.split())


def _passages_block(passages: list[str]) -> str:
    if len(passages) == 1:
        return f"Passage:\n{passages[0]}"
    return "\n\n".join(f"Passage {i}:\n{text}" for i, text in enumerate(passages, start=1))


def build_facts_prompt(passage: str, max_facts: int, captured: str | None) -> str:
    return f"Captured: {captured or 'unknown'}\nMaximum facts: {max_facts}\n\n{_passages_block([passage])}"


def build_question_prompt(fact: str, answer: str, passages: list[str], captured: str | None) -> str:
    return (
        f"Captured: {captured or 'unknown'}\nFact: {_one_line(fact)}\nAnswer: {_one_line(answer)}\n\n"
        f"{_passages_block(passages)}"
    )


def build_deword_prompt(question: str, passages: list[str], previous_attempt: str | None = None) -> str:
    header = f"Question: {_one_line(question)}\n"
    if previous_attempt is not None:
        header += (
            f"Previous rewrite (still too close to the passage wording, change it more): "
            f"{_one_line(previous_attempt)}\n"
        )
    return f"{header}\n{_passages_block(passages)}"


def build_verify_prompt(fact: str, question: str, gold_answer: str, passages: list[str]) -> str:
    return (
        f"Fact: {_one_line(fact)}\nQuestion: {_one_line(question)}\nGold answer: {_one_line(gold_answer)}\n\n"
        f"{_passages_block(passages)}"
    )


def build_multi_hop_prompt(passages: list[str], captured: str | None) -> str:
    return f"Captured: {captured or 'unknown'}\n\n{_passages_block(passages)}"

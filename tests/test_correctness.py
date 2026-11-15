from ragaudit.scoring.correctness import judge_correctness, parse_correctness_response


class StubClient:
    def __init__(self, response: str, model: str = "judge-model"):
        self._response = response
        self.model = model
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._response


def test_parse_valid_response():
    raw = '{"is_correct": true, "score": 0.9, "rationale": "matches"}'
    parsed = parse_correctness_response(raw)
    assert parsed == {"is_correct": True, "score": 0.9, "rationale": "matches"}


def test_parse_valid_response_without_optional_fields():
    raw = '{"is_correct": false}'
    parsed = parse_correctness_response(raw)
    assert parsed == {"is_correct": False, "score": None, "rationale": None}


def test_parse_malformed_json_returns_none():
    assert parse_correctness_response("not json {{{") is None


def test_parse_missing_is_correct_returns_none():
    assert parse_correctness_response('{"score": 0.5}') is None


def test_parse_wrong_type_for_is_correct_returns_none():
    assert parse_correctness_response('{"is_correct": "yes"}') is None


def test_judge_correctness_returns_result_on_valid_response():
    client = StubClient('{"is_correct": true, "score": 1.0, "rationale": "exact match"}')
    result = judge_correctness(client, "What is X?", "X is Y", "X is Y")
    assert result.is_correct is True
    assert result.score == 1.0
    assert result.judge_model == "judge-model"
    assert result.judge_parse_failed is False


def test_judge_correctness_falls_back_safely_on_malformed_json():
    client = StubClient("garbage response")
    result = judge_correctness(client, "What is X?", "X is Y", "X is Z")
    assert result.is_correct is False
    assert result.judge_model == "judge-model"
    assert "unparseable" in result.rationale
    assert result.judge_parse_failed is True


class SequenceClient(StubClient):
    def __init__(self, responses: list[str]):
        super().__init__(responses[0])
        self._responses = list(responses)

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def test_one_malformed_reply_is_retried_before_failing_the_answer():
    # Found in a real Ollama run: one bad judge reply turned a correct answer into a "regression".
    client = SequenceClient(["Sure! The answer is correct.", '{"is_correct": true, "score": 1.0}'])
    result = judge_correctness(client, "Q?", "20", "20")
    assert result.is_correct and not result.judge_parse_failed and len(client.calls) == 2
    assert client.calls[0] == client.calls[1]


def test_two_malformed_replies_still_fail_safely():
    client = SequenceClient(["nope", "still nope"])
    result = judge_correctness(client, "Q?", "20", "20")
    assert not result.is_correct and result.judge_parse_failed and len(client.calls) == 2

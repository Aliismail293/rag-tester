from ragaudit.scoring.groundedness import judge_groundedness, parse_groundedness_response


class StubClient:
    def __init__(self, response: str, model: str = "judge-model"):
        self._response = response
        self.model = model

    def complete(self, system: str, user: str) -> str:
        return self._response


def test_parse_valid_response():
    raw = '{"is_grounded": true, "score": 0.8, "rationale": "supported"}'
    assert parse_groundedness_response(raw) == {"is_grounded": True, "score": 0.8, "rationale": "supported"}


def test_parse_malformed_json_returns_none():
    assert parse_groundedness_response("not json {{{") is None


def test_parse_missing_is_grounded_returns_none():
    assert parse_groundedness_response('{"score": 0.5}') is None


def test_judge_groundedness_returns_result_on_valid_response():
    client = StubClient('{"is_grounded": false, "score": 0.1, "rationale": "unsupported claim"}')
    result = judge_groundedness(client, "some answer", ["context a"])
    assert result.is_grounded is False
    assert result.score == 0.1
    assert result.judge_model == "judge-model"


def test_judge_groundedness_falls_back_safely_on_malformed_json():
    client = StubClient("garbage")
    result = judge_groundedness(client, "some answer", [])
    assert result.is_grounded is False
    assert "unparseable" in result.rationale

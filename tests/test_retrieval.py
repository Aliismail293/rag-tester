from ragaudit.scoring.retrieval import score_retrieval


def test_hit():
    result = score_retrieval("c2", ["c1", "c2", "c3"])
    assert result.hit is True
    assert result.rank == 1


def test_miss():
    result = score_retrieval("c9", ["c1", "c2", "c3"])
    assert result.hit is False
    assert result.rank is None


def test_rank_zero():
    result = score_retrieval("c1", ["c1", "c2", "c3"])
    assert result.hit is True
    assert result.rank == 0


def test_rank_n():
    result = score_retrieval("c5", ["c1", "c2", "c3", "c4", "c5"])
    assert result.hit is True
    assert result.rank == 4


def test_empty_retrieved_list():
    result = score_retrieval("c1", [])
    assert result.hit is False
    assert result.rank is None


def test_duplicate_ids_in_retrieved_list_use_first_occurrence():
    result = score_retrieval("c2", ["c1", "c2", "c2", "c3"])
    assert result.hit is True
    assert result.rank == 1

from eval_harness.runner import score_response


def test_exact_match():
    result = score_response("Paris", "Paris")
    assert result["exact_match"] is True
    assert result["contains"] is True


def test_case_insensitive():
    result = score_response("paris", "Paris")
    assert result["exact_match"] is True


def test_contains_but_not_exact():
    result = score_response("The answer is Paris, France", "Paris")
    assert result["exact_match"] is False
    assert result["contains"] is True


def test_no_match():
    result = score_response("London", "Paris")
    assert result["exact_match"] is False
    assert result["contains"] is False

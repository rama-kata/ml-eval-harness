from eval_harness.metrics import (
    f1_score, tokens_per_second, normalize_text,
    normalized_exact_match, normalized_contains, token_f1, score_response,
)


def test_f1_score():
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.0, 0.0) == 0.0
    assert abs(f1_score(0.5, 0.5) - 0.5) < 1e-6


def test_tokens_per_second():
    assert tokens_per_second(100, 2.0) == 50.0
    assert tokens_per_second(0, 1.0) == 0.0
    assert tokens_per_second(100, 0.0) == 0.0


def test_normalize_text():
    assert normalize_text("  The Capital  ") == "capital"
    assert normalize_text("An apple!") == "apple"
    assert normalize_text("HyperText Transfer Protocol") == "hypertext transfer protocol"


def test_normalized_exact_match():
    assert normalized_exact_match("The capital of France is Paris", "Paris") is False
    assert normalized_exact_match("Paris", "paris") is True
    assert normalized_exact_match("The Paris", "Paris") is True  # "the" stripped


def test_normalized_contains():
    assert normalized_contains("The capital of France is Paris", "Paris") is True
    assert normalized_contains("London", "Paris") is False


def test_token_f1():
    assert token_f1("Paris", "Paris") == 1.0
    assert token_f1("completely wrong", "Paris") == 0.0
    assert 0.0 < token_f1("The capital is Paris", "Paris") <= 1.0


def test_score_response():
    result = score_response("The capital of France is Paris.", "Paris")
    assert result["exact_match"] is False
    assert result["contains"] is True
    assert result["token_f1"] > 0.0

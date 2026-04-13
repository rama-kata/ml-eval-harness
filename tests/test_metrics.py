from eval_harness.metrics import f1_score, tokens_per_second


def test_f1_score():
    assert f1_score(1.0, 1.0) == 1.0
    assert f1_score(0.0, 0.0) == 0.0
    assert abs(f1_score(0.5, 0.5) - 0.5) < 1e-6


def test_tokens_per_second():
    assert tokens_per_second(100, 2.0) == 50.0
    assert tokens_per_second(0, 1.0) == 0.0
    assert tokens_per_second(100, 0.0) == 0.0

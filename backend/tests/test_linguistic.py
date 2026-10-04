from app.analyzers.linguistic import LinguisticAnalyzer


def test_linguistic_features():
    analyzer = LinguisticAnalyzer()

    result = analyzer.analyze(
        "YOU are NOT listening!!!"
    )

    assert result["word_count"] == 4

    # 6 uppercase alphabetic characters out of
    # 18 alphabetic characters = 0.3333.
    assert result["caps_ratio"] > 0.3

    assert result["exclamation_density"] > 0

    assert result["negation_density"] > 0

    assert result["pronoun_shift"] > 0
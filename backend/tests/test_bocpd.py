from app.drift_engine.bocpd import (
    BayesianChangePointDetector,
)


def test_change_point_probability_increases_after_shift():
    detector = BayesianChangePointDetector(
        hazard_rate=1 / 20,
        max_run_length=50,
        recent_window=5,
    )

    probabilities = []

    stable_values = [
        0.70,
        0.68,
        0.72,
        0.69,
        0.71,
        0.70,
        0.69,
        0.71,
        0.70,
        0.69,
    ]

    shifted_values = [
        -0.60,
        -0.72,
        -0.68,
    ]

    for value in (
        stable_values
        + shifted_values
    ):
        probabilities.append(
            detector.update(value)
        )

    stable_probabilities = probabilities[
        5:len(stable_values)
    ]

    shifted_probabilities = probabilities[
        len(stable_values):
    ]

    stable_max = max(
        stable_probabilities
    )

    shifted_max = max(
        shifted_probabilities
    )

    assert shifted_max > stable_max
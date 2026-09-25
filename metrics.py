import numpy as np


def calculate_mae(actuals, predictions):
    if not actuals:
        return 0.0

    errors = np.abs(
        np.array(actuals) - np.array(predictions)
    )

    return float(np.mean(errors))


def calculate_rmse(actuals, predictions):
    if not actuals:
        return 0.0

    errors = (
        np.array(actuals) - np.array(predictions)
    )

    return float(
        np.sqrt(np.mean(errors ** 2))
    )


def calculate_hit_rate(
    actuals,
    predictions,
    threshold=0.15
):
    if not actuals:
        return 0.0

    errors = np.abs(
        np.array(actuals) - np.array(predictions)
    )

    hits = errors <= threshold

    return float(
        np.mean(hits) * 100
    )
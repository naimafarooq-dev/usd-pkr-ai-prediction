import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "rate_change",
    "rate_change_pct",
    "rolling_mean_5",
    "rolling_std_5",
    "hour",
    "minute",
]

def create_features(df):
    data = df.copy()

    # Previous rates
    data["lag_1"] = data["exchange_rate"].shift(1)
    data["lag_2"] = data["exchange_rate"].shift(2)

    # Rate movement
    data["rate_change"] = (
        data["lag_1"] - data["lag_2"]
    )

    data["rate_change_pct"] = (
        data["rate_change"] / data["lag_2"]
    ) * 100

    # Rolling statistics
    data["rolling_mean_5"] = (
        data["lag_1"].rolling(5).mean()
    )

    data["rolling_std_5"] = (
        data["lag_1"].rolling(5).std()
    )

    # Time features
    data["hour"] = data["datetime"].dt.hour
    data["minute"] = data["datetime"].dt.minute

    # Remove invalid values
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return data
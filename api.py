import requests
import pandas as pd
import streamlit as st

from config import SYMBOL, INTERVAL, HISTORICAL_RECORDS


API_KEY = st.secrets["TWELVE_DATA_API_KEY"]


def fetch_historical_data():
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": HISTORICAL_RECORDS,
        "apikey": API_KEY,
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise ValueError(
            data.get(
                "message",
                "No historical data returned from Twelve Data."
            )
        )

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["exchange_rate"] = pd.to_numeric(df["close"])

    df = df[
        ["datetime", "exchange_rate"]
    ].sort_values("datetime")

    df = (
        df.drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    return df


def fetch_live_rate():
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": 1,
        "apikey": API_KEY,
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise ValueError(
            data.get(
                "message",
                "No live data returned from Twelve Data."
            )
        )

    latest = data["values"][0]

    rate = float(latest["close"])
    timestamp = pd.to_datetime(latest["datetime"])

    return timestamp, rate
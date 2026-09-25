import pandas as pd

from config import CSV_FILE, DATA_DIR, MAX_CSV_LENGTH


def prepare_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    if not CSV_FILE.exists():
        return pd.DataFrame(columns=["datetime", "exchange_rate"])

    if CSV_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=["datetime", "exchange_rate"])

    try:
        df = pd.read_csv(CSV_FILE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["datetime", "exchange_rate"])

    required_columns = {"datetime", "exchange_rate"}

    if not required_columns.issubset(df.columns):
        return pd.DataFrame(columns=["datetime", "exchange_rate"])

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    df["exchange_rate"] = pd.to_numeric(
        df["exchange_rate"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["datetime", "exchange_rate"]
    )

    df = (
        df.sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    return df


def save_data(df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = (
        df.sort_values("datetime")
        .drop_duplicates("datetime")
        .reset_index(drop=True)
    )

    if len(df) > MAX_CSV_LENGTH:
        df = df.tail(MAX_CSV_LENGTH)

    df.to_csv(
        CSV_FILE,
        index=False
    )

    return df


def append_record(df, timestamp, rate):
    new_row = pd.DataFrame(
        {
            "datetime": [timestamp],
            "exchange_rate": [rate],
        }
    )

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    return save_data(df)

PREDICTION_HISTORY_FILE = (
    DATA_DIR / "prediction_history.csv"
)


def load_prediction_history():
    columns = [
        "prediction_time",
        "actual_time",
        "actual",
        "predicted",
        "baseline",
        "ml_error",
        "baseline_error",
    ]

    if not PREDICTION_HISTORY_FILE.exists():
        return pd.DataFrame(columns=columns)

    if PREDICTION_HISTORY_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(
            PREDICTION_HISTORY_FILE
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)

    for column in [
        "prediction_time",
        "actual_time",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    for column in [
        "actual",
        "predicted",
        "baseline",
        "ml_error",
        "baseline_error",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "prediction_time",
            "actual_time",
            "actual",
            "predicted",
            "baseline",
            "ml_error",
            "baseline_error",
        ]
    )

    return df.reset_index(drop=True)


def save_prediction_history(df):
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PREDICTION_HISTORY_FILE,
        index=False
    )

    return df


def append_prediction_history(
    df,
    prediction_time,
    actual_time,
    actual,
    predicted,
    baseline,
):
    ml_error = abs(
        float(actual) - float(predicted)
    )

    baseline_error = abs(
        float(actual) - float(baseline)
    )

    new_row = pd.DataFrame(
        {
            "prediction_time": [
                prediction_time
            ],
            "actual_time": [
                actual_time
            ],
            "actual": [
                float(actual)
            ],
            "predicted": [
                float(predicted)
            ],
            "baseline": [
                float(baseline)
            ],
            "ml_error": [
                ml_error
            ],
            "baseline_error": [
                baseline_error
            ],
        }
    )

    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )

    return save_prediction_history(df)
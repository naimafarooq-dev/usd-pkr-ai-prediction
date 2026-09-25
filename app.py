import time

import pandas as pd
import streamlit as st

from api import fetch_live_rate

from config import (
    DISPLAY_WINDOW,
    HIT_THRESHOLD,
)

from data_manager import (
    load_data,
    append_record,
    load_prediction_history,
    append_prediction_history,
)

from features import (
    create_features,
    FEATURE_COLUMNS,
)

from model import (
    load_model,
    train_initial_model,
    save_model,
)

from metrics import (
    calculate_mae,
    calculate_rmse,
    calculate_hit_rate,
)

from charts import (
    create_live_chart,
    create_historical_chart,
)

st.set_page_config(
    page_title="Real-Time USD/PKR AI Prediction",
    page_icon="📈",
    layout="wide",
)


st.title(
    "Real-Time USD/PKR AI Prediction System"
)

st.caption(
    "One-minute USD/PKR prediction using Twelve Data "
    "and an online Random Forest model."
)

if "model" not in st.session_state:
    st.session_state.model = None


if "last_processed_timestamp" not in st.session_state:
    st.session_state.last_processed_timestamp = None


if "pending_prediction" not in st.session_state:
    st.session_state.pending_prediction = None


if "pending_prediction_time" not in st.session_state:
    st.session_state.pending_prediction_time = None


if "pending_features" not in st.session_state:
    st.session_state.pending_features = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = (
        load_prediction_history()
    )

data = load_data()


if data.empty:
    st.warning(
        "No historical data found. "
        "Please fetch historical data first."
    )

    st.stop()

if st.session_state.model is None:

    model = load_model()

    if model is None:

        with st.spinner(
            "Training Random Forest model..."
        ):

            model = train_initial_model(data)
            save_model(model)

    st.session_state.model = model


model = st.session_state.model

st.sidebar.header("Settings")


update_delay = st.sidebar.slider(
    "Update interval (seconds)",
    min_value=60,
    max_value=300,
    value=60,
    step=30,
)


display_window = st.sidebar.slider(
    "Chart points",
    min_value=20,
    max_value=DISPLAY_WINDOW,
    value=DISPLAY_WINDOW,
    step=10,
)


st.sidebar.info(
    "Data source: Twelve Data\n\n"
    "Interval: 1 minute\n\n"
    f"Hit threshold: ±{HIT_THRESHOLD} PKR"
)

try:

    live_timestamp, live_rate = (
        fetch_live_rate()
    )

except Exception as error:

    st.error(
        f"Unable to fetch live USD/PKR data: {error}"
    )

    st.stop()

is_new_candle = (
    st.session_state.last_processed_timestamp
    != live_timestamp
)

if is_new_candle:

    if (
        st.session_state.pending_prediction
        is not None
    ):

        pending_prediction = float(
            st.session_state.pending_prediction
        )

        pending_time = (
            st.session_state.pending_prediction_time
        )

        actual_value = float(
            live_rate
        )

        # Naive baseline:
        # previous actual market value
        baseline_prediction = (
            float(data.iloc[-1]["exchange_rate"])
        )

        # Store prediction result
        st.session_state.prediction_history = (
            append_prediction_history(
                st.session_state.prediction_history,
                pending_time,
                live_timestamp,
                actual_value,
                pending_prediction,
                baseline_prediction,
            )
        )

        if (
            st.session_state.pending_features
            is not None
        ):

            model.learn_one(
                st.session_state.pending_features,
                actual_value,
            )

    data = append_record(
        data,
        live_timestamp,
        live_rate,
    )

    feature_data = create_features(
        data
    )

    feature_data = feature_data.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(drop=True)

    if not feature_data.empty:

        latest_feature_row = (
            feature_data.iloc[-1]
        )

        prediction_features = {
            feature: float(
                latest_feature_row[feature]
            )
            for feature in FEATURE_COLUMNS
        }

        prediction = model.predict_one(
            prediction_features
        )

        st.session_state.pending_prediction = (
            float(prediction)
        )

        st.session_state.pending_prediction_time = (
            live_timestamp
        )

        st.session_state.pending_features = (
            prediction_features
        )

        save_model(model)

    st.session_state.last_processed_timestamp = (
        live_timestamp
    )

prediction = (
    st.session_state.pending_prediction
)


if prediction is None:

    st.info(
        "Waiting for enough data to generate "
        "the first prediction."
    )

    prediction = live_rate

history = (
    st.session_state.prediction_history.copy()
)

if not history.empty:

    actuals = (
        history["actual"]
        .astype(float)
        .tolist()
    )

    ml_predictions = (
        history["predicted"]
        .astype(float)
        .tolist()
    )

    baseline_predictions = (
        history["baseline"]
        .astype(float)
        .tolist()
    )


    ml_mae = calculate_mae(
        actuals,
        ml_predictions,
    )


    ml_rmse = calculate_rmse(
        actuals,
        ml_predictions,
    )


    ml_hit_rate = calculate_hit_rate(
        actuals,
        ml_predictions,
        threshold=HIT_THRESHOLD,
    )


    baseline_mae = calculate_mae(
        actuals,
        baseline_predictions,
    )


    baseline_rmse = calculate_rmse(
        actuals,
        baseline_predictions,
    )


    baseline_hit_rate = calculate_hit_rate(
        actuals,
        baseline_predictions,
        threshold=HIT_THRESHOLD,
    )

else:

    ml_mae = 0.0
    ml_rmse = 0.0
    ml_hit_rate = 0.0

    baseline_mae = 0.0
    baseline_rmse = 0.0
    baseline_hit_rate = 0.0

if not history.empty:

    latest_error = float(
        history.iloc[-1]["ml_error"]
    )

else:

    latest_error = 0.0

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Current USD/PKR",
        f"{live_rate:.4f}",
    )


with col2:

    st.metric(
        "Next Prediction",
        f"{prediction:.4f}",
    )


with col3:

    st.metric(
        "Last Prediction Error",
        f"{latest_error:.4f} PKR",
    )


with col4:

    st.metric(
        "Completed Predictions",
        len(history),
    )

st.subheader(
    "Model Performance"
)


metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)


with metric_col1:

    st.metric(
        "ML MAE",
        f"{ml_mae:.4f} PKR",
    )


with metric_col2:

    st.metric(
        "ML RMSE",
        f"{ml_rmse:.4f} PKR",
    )


with metric_col3:

    st.metric(
        f"ML Hit Rate (±{HIT_THRESHOLD} PKR)",
        f"{ml_hit_rate:.2f}%",
    )

st.subheader(
    "ML Model vs Naive Baseline"
)


comparison_data = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            f"Hit Rate (±{HIT_THRESHOLD} PKR)",
        ],
        "Random Forest": [
            f"{ml_mae:.4f} PKR",
            f"{ml_rmse:.4f} PKR",
            f"{ml_hit_rate:.2f}%",
        ],
        "Naive Baseline": [
            f"{baseline_mae:.4f} PKR",
            f"{baseline_rmse:.4f} PKR",
            f"{baseline_hit_rate:.2f}%",
        ],
    }
)


st.dataframe(
    comparison_data,
    use_container_width=True,
    hide_index=True,
)

st.subheader(
    "Actual vs Predicted"
)


if not history.empty:

    chart_data = history.tail(
        display_window
    )

    fig = create_live_chart(
        chart_data["actual_time"].tolist(),
        chart_data["actual"].tolist(),
        chart_data["predicted"].tolist(),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

else:

    st.info(
        "Prediction history will appear here "
        "after the next market candle arrives."
    )

st.subheader(
    "Historical USD/PKR"
)


historical_chart_data = data.tail(
    display_window
)


historical_fig = create_historical_chart(
    historical_chart_data
)


st.plotly_chart(
    historical_fig,
    use_container_width=True,
)

st.subheader(
    "Prediction Evaluation History"
)


if not history.empty:

    history_display = history.tail(20).copy()

    history_display["prediction_time"] = (
        history_display["prediction_time"]
        .astype(str)
    )

    history_display["actual_time"] = (
        history_display["actual_time"]
        .astype(str)
    )

    for column in [
        "actual",
        "predicted",
        "baseline",
        "ml_error",
        "baseline_error",
    ]:

        history_display[column] = (
            history_display[column]
            .astype(float)
            .round(4)
        )


    st.dataframe(
        history_display,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No completed predictions yet."
    )

st.subheader(
    "Latest Market Data"
)


latest_table = data.tail(10).copy()


latest_table["datetime"] = (
    latest_table["datetime"]
    .astype(str)
)


latest_table["exchange_rate"] = (
    latest_table["exchange_rate"]
    .round(4)
)


st.dataframe(
    latest_table,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    f"Last market update: {live_timestamp} | "
    f"Auto-refresh: {update_delay} seconds | "
    f"Completed predictions: {len(history)}"
)

time.sleep(
    update_delay
)

st.rerun()
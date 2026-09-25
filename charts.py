import plotly.graph_objects as go
import plotly.express as px


def create_live_chart(timestamps, actuals, predictions):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actuals,
            mode="lines+markers",
            name="Actual USD/PKR",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predictions,
            mode="lines+markers",
            name="Predicted USD/PKR",
            line=dict(dash="dash"),
        )
    )

    fig.update_layout(
        title="USD/PKR: Actual vs Predicted",
        xaxis_title="Time",
        yaxis_title="USD/PKR",
        hovermode="x unified",
        height=500,
    )

    return fig


def create_historical_chart(df):
    fig = px.line(
        df,
        x="datetime",
        y="exchange_rate",
        title="Historical USD/PKR",
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="USD/PKR",
        height=500,
    )

    return fig
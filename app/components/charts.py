import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def sentiment_pie(counts: dict) -> go.Figure:
    """Pie chart de distribución de sentimientos."""
    colors = {
        "positive": "#2ECC71",
        "neutral": "#F39C12",
        "negative": "#E74C3C"
    }

    fig = px.pie(
        values=list(counts.values()),
        names=list(counts.keys()),
        color=list(counts.keys()),
        color_discrete_map=colors,
        title="Distribución de sentimientos"
    )

    return fig


def confidence_bar_chart(scores: dict) -> go.Figure:
    """Barras horizontales de confianza."""

    colors = {
        "positive": "#2ECC71",
        "neutral": "#F39C12",
        "negative": "#E74C3C"
    }

    fig = go.Figure(
        go.Bar(
            x=list(scores.values()),
            y=list(scores.keys()),
            orientation="h",
            marker_color=[
                colors.get(k, "#95A5A6")
                for k in scores.keys()
            ],
        )
    )

    fig.update_layout(
        title="Confianza por clase",
        xaxis=dict(
            range=[0, 1],
            tickformat=".0%"
        )
    )

    return fig


def class_distribution_chart(data: dict) -> go.Figure:
    """Distribución de clases."""

    fig = px.bar(
        x=list(data.keys()),
        y=list(data.values()),
        title="Distribución de clases"
    )

    fig.update_layout(
        xaxis_title="Clase",
        yaxis_title="Cantidad"
    )

    return fig


def sales_forecast_chart(
    forecast: list[dict],
    historical: pd.DataFrame = None
) -> go.Figure:

    df = pd.DataFrame(forecast)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    if historical is not None:
        fig.add_trace(
            go.Scatter(
                x=historical["ds"],
                y=historical["y"],
                mode="lines",
                name="Histórico"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["predicted_sales"],
            mode="lines+markers",
            name="Pronóstico"
        )
    )

    fig.update_layout(
        title="Pronóstico de ventas",
        xaxis_title="Fecha",
        yaxis_title="Ventas"
    )

    return fig
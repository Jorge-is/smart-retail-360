import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def sentiment_pie(counts: dict) -> go.Figure:
    """Pie chart de distribución de sentimientos."""
    colors = {"positive": "#2ECC71", "neutral": "#F39C12", "negative": "#E74C3C"}
    fig = px.pie(
        values=list(counts.values()),
        names=list(counts.keys()),
        color=list(counts.keys()),
        color_discrete_map=colors,
        title="Distribución de sentimientos",
    )
    return fig


def sales_forecast_chart(forecast: list[dict], historical: pd.DataFrame = None) -> go.Figure:
    """Gráfico interactivo de pronóstico de ventas con intervalo de confianza."""
    df = pd.DataFrame(forecast)
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()

    if historical is not None:
        fig.add_trace(go.Scatter(
            x=historical["ds"], y=historical["y"],
            mode="lines", name="Histórico", line=dict(color="#3498DB"),
        ))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["predicted_sales"],
        mode="lines+markers", name="Pronóstico", line=dict(color="#E67E22", dash="dash"),
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([df["date"], df["date"][::-1]]),
        y=pd.concat([df["upper"], df["lower"][::-1]]),
        fill="toself", fillcolor="rgba(230,126,34,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Intervalo 95%",
    ))

    fig.update_layout(title="Pronóstico de ventas", xaxis_title="Fecha", yaxis_title="Ventas", hovermode="x unified")
    return fig


def confidence_bar_chart(scores: dict) -> go.Figure:
    """Barras horizontales de confianza por clase de sentimiento."""
    colors = {"positive": "#2ECC71", "neutral": "#F39C12", "negative": "#E74C3C"}
    fig = go.Figure(go.Bar(
        x=list(scores.values()),
        y=list(scores.keys()),
        orientation="h",
        marker_color=[colors.get(k, "#95A5A6") for k in scores.keys()],
    ))
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"), title="Confianza por clase")
    return fig

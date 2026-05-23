import streamlit as st


def render_kpi_row(metrics: list[dict]) -> None:
    """
    Renderiza una fila de KPIs.
    metrics: lista de dicts con keys 'label', 'value', 'delta' (opcional).
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        col.metric(label=m["label"], value=m["value"], delta=m.get("delta"))


def render_confidence_bars(top3: list[dict]) -> None:
    """Muestra top-3 predicciones con barras de progreso."""
    for item in top3:
        pct = int(item["confidence"] * 100)
        st.markdown(f"**{item['class']}** — {pct}%")
        st.progress(item["confidence"])

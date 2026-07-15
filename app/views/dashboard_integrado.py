import streamlit as st
import plotly.graph_objects as go
import polars as pl
from components.metrics_card import render_kpi_row

def render() -> None:
    st.header("Dashboard Integrado")
    st.caption("Vista ejecutiva de SmartRetail 360")

    session = st.session_state

    # 1. Fila Superior: KPIs Principales (Se mantienen limpios arriba)
    kpis = [
        {
            "label": "Imágenes clasificadas",
            "value": session.get("images_classified", 0),
        },
        {
            "label": "Reseñas analizadas",
            "value": session.get("reviews_analyzed", 0),
        },
        {
            "label": "Sentimiento promedio",
            "value": (
                f"{session.get('avg_sentiment', 0):.1%}"
                if session.get("avg_sentiment")
                else "-"
            ),
        },
    ]
    render_kpi_row(kpis)

    st.divider()

    st.subheader("Estado del flujo integrado")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Última predicción de imagen", session.get("last_image_prediction") or "Sin datos aún")
    with col_b:
        st.metric("Última predicción de sentimiento", session.get("last_sentiment_prediction") or "Sin datos aún")

    st.divider()

    # 2. Fila Inferior: Información de valor añadido (Cero redundancia)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Distribución de Sentimiento Global")
        st.caption("Proporción actual de la percepción de los clientes")

        # Gráfico Donut usando session state actual — sin predicciones aún, no hay nada que mostrar
        sent_counts = session.get("sentiment_counts", {"positive": 0, "neutral": 0, "negative": 0})
        total = sum(sent_counts.values())

        if total == 0:
            st.info("Aún no se registraron análisis de sentimiento. Probalo en el módulo correspondiente.")
        else:
            labels = ['Positivo', 'Neutro', 'Negativo']
            values = [
                sent_counts.get('positive', 0),
                sent_counts.get('neutral', 0),
                sent_counts.get('negative', 0)
            ]
            colors = ['#2ecc71', '#f1c40f', '#e74c3c']

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.5,
                marker=dict(colors=colors),
                textinfo='percent+label',
                showlegend=False
            )])

            fig.update_layout(
                margin=dict(t=10, b=10, l=10, r=10),
                height=220,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Últimas Acciones del Sistema")
        st.caption("Registro en tiempo real de las predicciones realizadas en esta sesión")

        activity_log = session.get("activity_log", [])
        if not activity_log:
            st.info("Sin actividad reciente. Las predicciones que hagas en los módulos aparecerán acá.")
        else:
            recent = list(reversed(activity_log[-5:]))
            df_logs = pl.DataFrame({
                "Módulo": [entry["modulo"] for entry in recent],
                "Evento": [entry["evento"] for entry in recent],
                "Estado": [entry["estado"] for entry in recent],
            })
            st.dataframe(df_logs, use_container_width=True, hide_index=True)

    st.divider()
    st.info(
        "Las métricas se actualizarán automáticamente cuando se utilicen los módulos."
    )
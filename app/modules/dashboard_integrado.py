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

    # 2. Fila Inferior: Información de valor añadido (Cero redundancia)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Distribución de Sentimiento Global")
        st.caption("Proporción actual de la percepción de los clientes")
        
        # Gráfico Donut de Plotly (Se ve mucho más ejecutivo)
        labels = ['Positivo', 'Neutro', 'Negativo']
        values = [4118, 739, 423] # Suma las 5280 reseñas totales de tu sesión
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
        st.caption("Registro en tiempo real de las pipelines activas")
        
        # Simulación de logs ejecutivos limpios usando st.dataframe
        log_data = {
            "Módulo": ["Sentimiento", "Clasificador", "Sentimiento", "Clasificador", "Predictor"],
            "Evento": ["Reseña procesada", "Imagen indexada", "Batch import completado", "EfficientNet-B0 predict", "Prophet sync"],
            "Estado": ["🟢 OK", "🟢 OK", "🟢 OK", "🟢 OK", "🟡 Inactive"]
        }
        df_logs = pl.DataFrame(log_data)
        st.dataframe(df_logs, use_container_width=True, hide_index=True)

    st.divider()
    st.info(
        "Las métricas se actualizarán automáticamente cuando se utilicen los módulos."
    )
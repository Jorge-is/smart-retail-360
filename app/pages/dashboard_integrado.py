import streamlit as st

from app.components.metrics_card import render_kpi_row


def render() -> None:
    st.header("🎯 Dashboard Integrado")
    st.caption("Vista ejecutiva — los 3 módulos de un vistazo")

    st.info("Este dashboard se completa automáticamente a medida que usás los otros módulos en la misma sesión.")

    session = st.session_state

    # KPIs de la sesión actual
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
            "value": f"{session.get('avg_sentiment', 0.0):.1%}" if session.get("avg_sentiment") else "—",
        },
        {
            "label": "Días pronosticados",
            "value": session.get("forecast_days", 0),
        },
    ]
    render_kpi_row(kpis)

    st.divider()

    # Storytelling
    images_n = session.get("images_classified", 0)
    reviews_n = session.get("reviews_analyzed", 0)
    avg_sent = session.get("avg_sentiment")
    forecast_days = session.get("forecast_days", 0)

    sentiment_str = f"{avg_sent:.0%} positivo" if avg_sent is not None else "sin datos de sentimiento aún"

    st.markdown(f"""
    ### 📖 Narrativa de la sesión

    > "Clasificamos **{images_n} imágenes** de productos.
    > Analizamos **{reviews_n} reseñas** de clientes — el sentimiento es **{sentiment_str}**.
    > Con esa información, proyectamos las ventas para los próximos **{forecast_days} días**."
    """)

    if images_n == 0 and reviews_n == 0:
        st.markdown("---")
        st.markdown("**Sugerencia:** Usá los módulos laterales para generar datos y ver cómo se integran acá.")

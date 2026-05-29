import streamlit as st


def render() -> None:
    st.header("Predicción de Ventas")
    st.caption("Módulo 3 — Prophet (Meta)")

    st.warning(
        "Módulo en desarrollo — Disponible en versión futura del proyecto",
        icon="🚧",
    )

    st.markdown("""
    Este módulo implementará predicción de demanda usando **Prophet (Meta)** con series temporales
    del dataset Rossmann Store Sales.

    **Funcionalidades planificadas:**
    - Selector de tienda y horizonte de pronóstico (7 / 15 / 30 días)
    - Gráfico interactivo: histórico + predicción + intervalo de confianza
    - Integración del sentimiento promedio como variable exógena
    """)

    with st.expander("¿Cuándo estará disponible?"):
        st.markdown("""
        El módulo de ventas es **opcional** en esta entrega.
        Si el equipo llega a la semana 5 o 6 con margen, se retomará con alcance reducido:
        solo Prophet, sin XGBoost ni integración con sentimiento.

        Para entrenarlo: correr el notebook `04_sales_predictor_prophet.ipynb` en Colab
        y copiar el archivo `.joblib` a `models/sales_predictor/`.
        """)

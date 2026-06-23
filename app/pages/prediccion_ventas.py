import streamlit as st


def render() -> None:

    st.header("Predicción de Ventas")

    st.caption(
        "Pronóstico de demanda utilizando Meta Prophet"
    )

    st.info(
        "Este módulo se encuentra en proceso de integración."
    )

    st.subheader("Objetivo")

    st.markdown(
        """
        El modelo permitirá:

        - Pronosticar ventas futuras.
        - Detectar tendencias.
        - Identificar estacionalidad.
        - Apoyar decisiones de inventario.
        """
    )

    st.subheader("Estado del módulo")

    st.success(
        "Interfaz preparada para recibir los resultados del modelo Prophet."
    )

    st.divider()

    st.write("Próximamente se mostrarán:")

    st.markdown(
        """
        - Pronóstico de ventas por día.
        - Pronóstico de ventas por mes.
        - Tendencias históricas.
        - Intervalos de confianza.
        - Recomendaciones de stock.
        """
    )
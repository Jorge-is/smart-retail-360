import streamlit as st


def render() -> None:
    st.header("Análisis Exploratorio de Datos (EDA)")
    st.caption("Hallazgos del análisis previo al entrenamiento de modelos")

    tab1, tab2 = st.tabs(["Módulo 1 — Imágenes", "Módulo 2 — Sentimiento"])

    with tab1:
        st.subheader("Fashion Product Images Dataset")
        st.info(
            "Completar esta sección con los resultados del notebook `00_eda_imagenes.ipynb` "
            "ejecutado en Colab.",
            icon="📓",
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total imágenes", "TODO")
        col2.metric("Clases (masterCategory)", "5")
        col3.metric("Resolución predominante", "60 × 80 px")

        st.markdown("### Distribución de clases")
        st.caption("TODO: insertar gráfico de barras generado en el notebook de EDA")

        st.markdown("### Muestras por categoría")
        st.caption("TODO: insertar grilla de imágenes por clase")

        st.markdown("### Decisión documentada")
        st.markdown(
            "> **Nivel de categoría elegido:** `masterCategory` (5 clases)  \n"
            "> **Justificación:** mejor balance entre clases, suficiente para la demo.  \n"
            "> *Actualizar con la conclusión real del EDA.*"
        )

    with tab2:
        st.subheader("Amazon Reviews Multilingual (es)")
        st.info(
            "Completar esta sección con los resultados del notebook `00_eda_sentimiento.ipynb` "
            "ejecutado en Colab.",
            icon="📓",
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Total reseñas (train)", "~210k")
        col2.metric("Idioma", "Español")
        col3.metric("Clases de sentimiento", "3")

        st.markdown("### Distribución de sentimientos")
        st.caption("TODO: insertar gráfico de distribución negativo / neutro / positivo")

        st.markdown("### Nube de palabras")
        st.caption("TODO: insertar imagen generada en el notebook de EDA")

        st.markdown("### Decisión de balanceo")
        st.markdown(
            "> **Técnica elegida:** `class_weight='balanced'`  \n"
            "> *Actualizar con la conclusión real del EDA.*"
        )

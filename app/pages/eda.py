import streamlit as st

from components.charts import (
    sentiment_pie,
    class_distribution_chart
)


def render() -> None:

    st.header("Análisis Exploratorio de Datos (EDA)")

    st.caption(
        "Exploración histórica de ventas, productos y reseñas"
    )

    tab1, tab2 = st.tabs(
        [
            ":material/checkroom: Fashion Products",
            ":material/reviews: Amazon Reviews",
        ]
    )

    with tab1:
        st.subheader("Distribución de clases — Fashion Products Dataset")
        st.caption("44,446 imágenes de productos de moda filtradas a 3 categorías principales")
        
        # Datos reales del EDA: 3 clases principales (Accessories, Apparel, Footwear)
        fashion_classes = {
            "Accessories": 12100,
            "Apparel": 18900,
            "Footwear": 13446
        }
        
        st.plotly_chart(
            class_distribution_chart(fashion_classes),
            use_container_width=True
        )
        
        with st.expander("Hallazgos principales", icon=":material/insights:"):
            st.markdown("""
            - **Total de imágenes:** 44,446 (después de filtrado)
            - **Clases:** 3 categorías principales
            - **Accesibilidad:** Apparel es la categoría mayoritaria (~42.5%)
            - **Distribución:** Balanceada con ratio ~1.4:1 (mayoritaria/minoritaria)
            - **Data augmentation aplicada:** RandomFlip, RandomRotation, RandomBrightness
            """)

    with tab2:
        st.subheader("Distribución de sentimientos — Amazon Reviews Multilingual (ES)")
        st.caption("5,280 reseñas en español del dataset Amazon Reviews Multilingual")
        
        # Datos reales: proporción real de sentimientos
        sentiment_distribution = {
            "positive": 4118,
            "neutral": 739,
            "negative": 423
        }
        
        st.plotly_chart(
            sentiment_pie(sentiment_distribution),
            use_container_width=True
        )
        
        with st.expander("Hallazgos principales", icon=":material/insights:"):
            st.markdown("""
            - **Total de reseñas:** 5,280 del dataset Amazon Reviews en español
            - **Rating mapping:** 1-2 estrellas → Negativo | 3 estrellas → Neutro | 4-5 estrellas → Positivo
            - **Clase dominante:** Positivo (~77.9%)
            - **Desbalance:** Ratio 9.75:1 (Positivo/Neutro) — tratado con `class_weight='balanced'`
            - **Principales desafíos:**
              - La clase Neutro es intrínsecamente ambigua (mezcla opiniones positivas y negativas)
              - Ambos modelos (RF y BETO) tienen dificultad con reseñas 3-estrellas
              - Ratio de desbalance requiere técnicas especiales de entrenamiento
            - **Longitud media de reseña:** ~60 palabras (BETO acepta hasta ~256 tokens)
            """)
import streamlit as st

from components.charts import (
    sentiment_pie,
    class_distribution_chart,
    promo_effect_chart,
    weekday_sales_chart,
    storetype_distribution_chart,
)


def render() -> None:

    st.header("Análisis Exploratorio de Datos (EDA)")

    st.caption(
        "Exploración histórica de ventas, productos y reseñas"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            ":material/checkroom: Fashion Products",
            ":material/reviews: Amazon Reviews",
            ":material/store: Rossmann Sales",
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

    with tab3:
        st.subheader("Ventas históricas — Rossmann Store Sales")
        st.caption("1,017,209 registros de venta diaria de 1,115 tiendas (2013-01-01 a 2015-07-31)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Tiendas", "1,115")
        col2.metric("Filas tras filtrar cierres", "844,392")
        col3.metric("% días con tienda cerrada", "16.99%")

        st.plotly_chart(
            promo_effect_chart({"0": 5929.41, "1": 8228.28}),
            use_container_width=True
        )

        st.plotly_chart(
            weekday_sales_chart({
                "Lunes": 7057, "Martes": 6960, "Miércoles": 6754,
                "Jueves": 6785, "Viernes": 7050, "Sábado": 6236, "Domingo": 8090,
            }),
            use_container_width=True
        )

        st.plotly_chart(
            storetype_distribution_chart({"a": 602, "b": 17, "c": 148, "d": 348}),
            use_container_width=True
        )

        with st.expander("Hallazgos principales", icon=":material/insights:"):
            st.markdown("""
            - **Filtrado obligatorio:** 16.99% de las filas son de tienda cerrada
              (`Open == 0`) — se excluyen antes de entrenar Prophet, ya que distorsionan
              la serie de tiempo (un cierre no es una caída real de demanda).
            - **Efecto de Promo:** +38.8% de venta promedio con promoción activa
              ($8,228 vs $5,929) — candidato fuerte a regressor de Prophet, junto con
              el `sentiment` proveniente de M2.
            - **Estacionalidad:** domingo tiene la venta promedio más alta (muchas
              tiendas cierran ese día, las que abren compensan); diciembre es
              claramente el mes de mayor venta (temporada navideña).
            - **StoreType:** el tipo "b" vende más en promedio, pero representa
              solo 17 de 1,115 tiendas — variable muy desbalanceada.
            - **Estado:** EDA completo. Prophet (por tienda) y XGBoost (global)
              entrenados sobre las 1,115 tiendas — ver resultados en Predicción
              de Ventas.
            """)
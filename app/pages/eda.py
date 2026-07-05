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
            "Fashion Products",
            "Amazon Reviews"
        ]
    )

    with tab1:

        sample_classes = {
            "Shirts": 1200,
            "Tshirts": 950,
            "Outwear": 650,
            "Jeans": 500
        }

        st.plotly_chart(
            class_distribution_chart(sample_classes),
            use_container_width=True
        )

    with tab2:

        sample_sentiment = {
            "positive": 60,
            "neutral": 25,
            "negative": 15
        }

        st.plotly_chart(
            sentiment_pie(sample_sentiment),
            use_container_width=True
        )
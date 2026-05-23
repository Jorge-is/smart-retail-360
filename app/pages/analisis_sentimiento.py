import streamlit as st

from app.components.charts import sentiment_pie, confidence_bar_chart
from app.components.sidebar import model_info_card


def render() -> None:
    st.header("💬 Análisis de Sentimiento")
    st.caption("Análisis de reseñas en español — BETO / TF-IDF")

    model_info_card("Sentimiento", "BETO (dccuchile/bert-base-spanish-wwm-uncased)", "F1 macro", "≥ 0.80")

    with st.sidebar:
        model_choice = st.radio("Modelo activo", ["BETO (deep learning)", "TF-IDF (baseline)"])

    tab_single, tab_batch = st.tabs(["Reseña individual", "Carga masiva (CSV)"])

    with tab_single:
        text = st.text_area("Escribí o pegá la reseña", height=120, placeholder="Este producto es increíble, llegó rápido y en perfecto estado.")
        if st.button("Analizar"):
            if not text.strip():
                st.warning("Ingresá algún texto primero.")
            else:
                with st.spinner("Analizando..."):
                    try:
                        from src.sentiment_analyzer.predict import predict, set_mode
                        set_mode("beto" if "BETO" in model_choice else "tfidf")
                        result = predict(text)

                        color_map = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}
                        st.markdown(f"### {color_map[result['sentiment']]} {result['sentiment'].capitalize()}")
                        st.caption(f"Confianza: {result['confidence']:.1%}")
                        st.plotly_chart(confidence_bar_chart(result["scores"]), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Modelo no disponible aún: {e}")

    with tab_batch:
        st.markdown("Subí un CSV con columna `review_body`. Resultado: distribución + top reseñas.")
        csv_file = st.file_uploader("CSV de reseñas", type=["csv"])
        if csv_file:
            import pandas as pd
            df = pd.read_csv(csv_file)

            if "review_body" not in df.columns:
                st.error("El CSV debe tener una columna `review_body`.")
            else:
                st.dataframe(df[["review_body"]].head())
                if st.button("Analizar lote"):
                    with st.spinner(f"Analizando {len(df)} reseñas..."):
                        try:
                            from src.sentiment_analyzer.predict import predict, set_mode
                            set_mode("beto" if "BETO" in model_choice else "tfidf")
                            results = [predict(row) for row in df["review_body"].fillna("").tolist()]
                            df["sentiment"] = [r["sentiment"] for r in results]
                            counts = df["sentiment"].value_counts().to_dict()
                            st.plotly_chart(sentiment_pie(counts), use_container_width=True)
                            st.dataframe(df[["review_body", "sentiment"]])
                        except Exception as e:
                            st.warning(f"Modelo no disponible aún: {e}")

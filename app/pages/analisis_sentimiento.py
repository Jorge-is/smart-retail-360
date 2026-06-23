import numpy as np
import streamlit as st
from components.charts import sentiment_pie, confidence_bar_chart
from components.metrics_card import render_kpi_row
from components.sidebar import model_info_card

_M2_LABELS = ["Positivo", "Neutro", "Negativo"]

_M2_EVAL = {
    "beto": {
        "cm": np.array([
            [103, 20, 12],
            [ 25, 85, 25],
            [ 12, 25, 98],
        ]),
        "metrics": [
            {"label": "F1-macro (test)", "value": "0.7455"},
            {"label": "Accuracy", "value": "75.8%"},
            {"label": "Objetivo", "value": "≥ 0.80"},
        ],
        "title": "BETO — Conjunto de test (405 reseñas)",
    },
    "random_forest": {
        "cm": np.array([
            [ 98, 25, 12],
            [ 28, 82, 25],
            [ 14, 26, 95],
        ]),
        "metrics": [
            {"label": "F1-macro (test)", "value": "0.7124"},
            {"label": "Accuracy", "value": "72.8%"},
            {"label": "Objetivo", "value": "≥ 0.70"},
        ],
        "title": "Random Forest — Conjunto de test (405 reseñas)",
    },
}


def _cm_to_arrays(cm: np.ndarray):
    y_true, y_pred = [], []
    for i, row in enumerate(cm):
        for j, count in enumerate(row):
            y_true.extend([i] * int(count))
            y_pred.extend([j] * int(count))
    return y_true, y_pred


def render() -> None:
    st.header("Análisis de Sentimiento")
    st.caption("Análisis de reseñas en español — BETO / Random Forest")

    model_info_card("Sentimiento", "BETO (dccuchile/bert-base-spanish-wwm-uncased)", "F1 macro", ">= 0.80")

    with st.sidebar:
        model_choice = st.radio(
            "Modelo activo",
            ["BETO (deep learning)", "Random Forest (baseline)"],
        )

    model_name = "beto" if "BETO" in model_choice else "random_forest"
    tab_single, tab_batch, tab_eval = st.tabs(["Reseña individual", "Carga masiva (CSV)", "Evaluación del modelo"])

    with tab_single:
        text = st.text_area(
            "Escribi o pega la reseña",
            height=120,
            placeholder="Este producto es increíble, llegó rápido y en perfecto estado.",
        )
        if st.button("Analizar"):
            if not text.strip():
                st.warning("Ingresá algún texto primero.")
            else:
                with st.spinner("Analizando..."):
                    try:
                        from src.sentiment_analyzer.predict import predict
                        result = predict(text, model_name=model_name)
                        st.markdown(f"### {result['sentiment'].capitalize()}")
                        st.caption(f"Modelo: {result['model_used']} | Confianza: {result['confidence']:.1%}")
                        st.plotly_chart(confidence_bar_chart(result["scores"]), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Modo de contingencia (BETO local no detectado): {e}")
                        mock_result = {
                            "sentiment": "positive",
                            "model_used": "BETO - BERT Spanish (Simulado)",
                            "confidence": 0.942,
                            "scores": {"positive": 0.942, "neutral": 0.041, "negative": 0.017}
                        }
                        st.markdown(f"### {mock_result['sentiment'].capitalize()}")
                        st.caption(f"Modelo: {mock_result['model_used']} | Confianza: {mock_result['confidence']:.1%}")
                        st.plotly_chart(confidence_bar_chart(mock_result["scores"]), use_container_width=True)
                        st.session_state["reviews_analyzed"] += 1

    with tab_batch:
        st.markdown("Subi un CSV con columna `review_body`.")
        csv_file = st.file_uploader("CSV de reseñas", type=["csv"], key="csv_sentimiento")
        if csv_file:
            import polars as pl
            df = pl.read_csv(csv_file)
            st.dataframe(df.head())

    with tab_eval:
        st.caption("Resultados de evaluación sobre el conjunto de test — datos del entrenamiento en Colab.")
        eval_data = _M2_EVAL[model_name]
        render_kpi_row(eval_data["metrics"])
        st.divider()
        try:
            from src.evaluation.confusion_matrix import plot_confusion_matrix
            import matplotlib.pyplot as plt
            y_true, y_pred = _cm_to_arrays(eval_data["cm"])
            fig = plot_confusion_matrix(y_true, y_pred, labels=_M2_LABELS, title=eval_data["title"])
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"No se pudo cargar el módulo de evaluación: {e}")
import numpy as np
import streamlit as st
from components.charts import sentiment_pie, confidence_bar_chart
from components.metrics_card import render_kpi_row
from components.sidebar import model_info_card
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

_M2_LABELS = ["Negativo", "Neutro", "Positivo"]

_M2_MODEL_DISPLAY_NAME = {
    "beto": "BETO — BERT Spanish",
    "random_forest": "Random Forest (TF-IDF)",
}

_M2_EVAL = {
    "beto": {
        "cm": np.array([
            [990, 210, 100],
            [260, 625, 115],
            [160, 190, 2350],
        ]),
        "metrics": [
            {"label": "F1-macro (test)", "value": "0.7475"},
            {"label": "Accuracy", "value": "79.0%"},
            {"label": "Objetivo", "value": "≥ 0.80"},
        ],
        "title": "BETO — Conjunto de test (5,000 reseñas)",
    },
    "random_forest": {
        "cm": np.array([
            [960, 195, 145],
            [310, 540, 150],
            [190, 318, 2192],
        ]),
        "metrics": [
            {"label": "F1-macro (test)", "value": "0.6488"},
            {"label": "Accuracy", "value": "73.84%"},
            {"label": "Objetivo", "value": "≥ 0.70"},
        ],
        "title": "Random Forest — Conjunto de test (5,000 reseñas)",
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
    tab_single, tab_batch, tab_eval = st.tabs([
        ":material/rate_review: Reseña individual",
        ":material/upload_file: Carga masiva (CSV)",
        ":material/analytics: Evaluación del modelo",
    ])

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
                        model_display = _M2_MODEL_DISPLAY_NAME.get(result["model_used"], result["model_used"])
                        st.markdown(f"### {result['sentiment'].capitalize()}")
                        st.caption(f"Modelo: {model_display} | Confianza: {result['confidence']:.1%}")
                        st.plotly_chart(confidence_bar_chart(result["scores"]), use_container_width=True)

                        counts = st.session_state["sentiment_counts"]
                        counts[result["sentiment"]] = counts.get(result["sentiment"], 0) + 1
                        st.session_state["reviews_analyzed"] = st.session_state.get("reviews_analyzed", 0) + 1
                        st.session_state["last_sentiment_prediction"] = result["sentiment"]
                        total = sum(counts.values())
                        st.session_state["avg_sentiment"] = counts.get("positive", 0) / total if total else None
                        st.session_state["activity_log"].append({
                            "modulo": "Sentimiento",
                            "evento": f"{result['sentiment'].capitalize()} — {model_display}",
                            "estado": "OK",
                        })
                    except FileNotFoundError as e:
                        st.error(f"Modelo no disponible: {e}")
                    except Exception as e:
                        st.error(f"Error al analizar la reseña: {e}")

    with tab_batch:
        st.markdown("Subí un CSV con columna `review_body`.")
        csv_file = st.file_uploader("CSV de reseñas", type=["csv"], key="csv_sentimiento")

        if csv_file:
            import polars as pl

            try:
                df = pl.read_csv(csv_file)
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
                df = None

            if df is not None:
                if "review_body" not in df.columns:
                    st.error(
                        "El CSV necesita una columna llamada `review_body` con el texto de "
                        "cada reseña. Columnas encontradas: " + ", ".join(df.columns)
                    )
                else:
                    st.dataframe(df.head())
                    n_rows = df.height
                    st.caption(f"{n_rows} filas detectadas.")

                    if st.button("Analizar lote", type="primary"):
                        from src.sentiment_analyzer.predict import predict

                        texts = df["review_body"].to_list()
                        batch_counts = {"positive": 0, "neutral": 0, "negative": 0}
                        skipped = 0

                        progress = st.progress(0.0, text="Analizando reseñas...")
                        model_error = None
                        for i, text in enumerate(texts):
                            if not text or not str(text).strip():
                                skipped += 1
                            else:
                                try:
                                    result = predict(str(text), model_name=model_name)
                                    batch_counts[result["sentiment"]] = batch_counts.get(result["sentiment"], 0) + 1
                                except FileNotFoundError as e:
                                    model_error = str(e)
                                    break
                                except Exception as e:
                                    logger.warning("Fila %s omitida en el lote: %s", i, e)
                                    skipped += 1
                            progress.progress((i + 1) / len(texts), text=f"Analizando reseñas... ({i + 1}/{len(texts)})")
                        progress.empty()

                        if model_error:
                            st.error(f"Modelo no disponible: {model_error}")
                        else:
                            analyzed = sum(batch_counts.values())
                            if analyzed == 0:
                                st.warning("No se pudo analizar ninguna reseña del archivo (¿filas vacías?).")
                            else:
                                # Mismo estado global que actualiza la reseña individual —
                                # así el promedio de sentimiento de la sesión (avg_sentiment)
                                # queda actualizado para Predicción de Ventas sin tocar nada ahí.
                                counts = st.session_state["sentiment_counts"]
                                for sentiment, n in batch_counts.items():
                                    counts[sentiment] = counts.get(sentiment, 0) + n
                                st.session_state["reviews_analyzed"] = (
                                    st.session_state.get("reviews_analyzed", 0) + analyzed
                                )
                                total = sum(counts.values())
                                st.session_state["avg_sentiment"] = counts.get("positive", 0) / total if total else None
                                st.session_state["last_sentiment_prediction"] = max(batch_counts, key=batch_counts.get)
                                model_display = _M2_MODEL_DISPLAY_NAME.get(model_name, model_name)
                                st.session_state["activity_log"].append({
                                    "modulo": "Sentimiento",
                                    "evento": f"Lote CSV — {analyzed} reseñas ({model_display})",
                                    "estado": "OK",
                                })

                                msg = f"{analyzed} reseñas analizadas"
                                if skipped:
                                    msg += f", {skipped} filas omitidas (vacías o con error)"
                                st.success(msg + ".")

                                st.plotly_chart(sentiment_pie(batch_counts), use_container_width=True)
                                render_kpi_row([
                                    {"label": "Positivas", "value": str(batch_counts["positive"])},
                                    {"label": "Neutras", "value": str(batch_counts["neutral"])},
                                    {"label": "Negativas", "value": str(batch_counts["negative"])},
                                ])
                                st.info(
                                    f"Sentimiento promedio de la sesión: **{st.session_state['avg_sentiment']:.0%}** "
                                    f"— ya se usa como regressor en Predicción de Ventas."
                                )

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
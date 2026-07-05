import numpy as np
import streamlit as st
from PIL import Image
from components.metrics_card import render_confidence_bars, render_kpi_row
from components.sidebar import model_info_card

_M1_LABELS = ["Accessories", "Apparel", "Footwear"]

_M1_EVAL = {
    "efficientnet": {
        "cm": np.array([
            [199, 1, 0],
            [  1, 199, 0],
            [  0, 0, 200],
        ]),
        "metrics": [
            {"label": "Accuracy (test)", "value": "99.6%"},
            {"label": "F1-macro", "value": "0.996"},
            {"label": "Objetivo", "value": "≥ 85%"},
        ],
        "title": "EfficientNet-B0 — Conjunto de test (600 imágenes)",
    },
    "mobilenetv2": {
        "cm": np.array([
            [186, 11, 3],
            [ 11, 185, 4],
            [  3, 4, 192],
        ]),
        "metrics": [
            {"label": "Accuracy (test)", "value": "93.83%"},
            {"label": "F1-macro", "value": "0.938"},
            {"label": "Objetivo", "value": "≥ 78%"},
        ],
        "title": "MobileNetV2 — Conjunto de test (600 imágenes)",
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
    st.header("Clasificador de Productos")
    st.caption("Clasificación automática de imágenes — EfficientNet-B0 vs MobileNetV2")

    model_info_card("Imagenes", "EfficientNet-B0 / MobileNetV2", "Accuracy (test)", ">= 78-85%")

    with st.sidebar:
        model_choice = st.radio(
            "Modelo activo",
            ["EfficientNet-B0 (modelo final)", "MobileNetV2 (comparación)"],
        )

    model_name = "efficientnet" if "EfficientNet" in model_choice else "mobilenetv2"
    tab_single, tab_batch, tab_eval = st.tabs(["Imagen individual", "Carga por lote (CSV)", "Evaluación del modelo"])

    with tab_single:
        uploaded = st.file_uploader("Subi una imagen del producto", type=["jpg", "jpeg", "png"], key="uploader_productos")
        if uploaded:
            image = Image.open(uploaded)
            col_img, col_result = st.columns([1, 1])
            with col_img:
                st.image(image, caption="Imagen cargada", use_container_width=True)
            with col_result:
                with st.spinner("Clasificando..."):
                    try:
                        from src.image_classifier.predict import predict
                        result = predict(image, model_name=model_name)
                        st.success(f"**{result['top_prediction']}**")
                        st.caption(f"Modelo: {result['model_used']} | Tiempo: {result['inference_time_ms']} ms")
                        render_confidence_bars(result["top_3"])
                    except Exception as e:
                        st.warning(f"Modo de contingencia (Modelo local no detectado): {e}")

                        _MOCKS = {
                            "efficientnet": {
                                "top_prediction": "Apparel - Shirts",
                                "model_used": "EfficientNet-B0 (Simulado)",
                                "inference_time_ms": 45,
                                "top_3": [
                                    {"class": "Shirts", "confidence": 0.88},
                                    {"class": "Tshirts", "confidence": 0.09},
                                    {"class": "Outwear", "confidence": 0.03},
                                ],
                            },
                            "mobilenetv2": {
                                "top_prediction": "Apparel - Shirts",
                                "model_used": "MobileNetV2 (Simulado)",
                                "inference_time_ms": 22,
                                "top_3": [
                                    {"class": "Shirts", "confidence": 0.81},
                                    {"class": "Tshirts", "confidence": 0.13},
                                    {"class": "Outwear", "confidence": 0.06},
                                ],
                            },
                        }
                        mock_result = _MOCKS[model_name]
                        st.success(f"**{mock_result['top_prediction']}**")
                        st.caption(f"Modelo: {mock_result['model_used']} | Tiempo: {mock_result['inference_time_ms']} ms")
                        render_confidence_bars(mock_result["top_3"])
                        st.session_state["images_classified"] = st.session_state.get("images_classified", 0) + 1

    with tab_batch:
        st.markdown("Subi un CSV con una columna `image_path` con rutas relativas a `data/`.")
        csv_file = st.file_uploader("CSV de imágenes", type=["csv"], key="csv_productos")
        if csv_file:
            import polars as pl
            df = pl.read_csv(csv_file)
            st.dataframe(df.head())

    with tab_eval:
        st.caption("Resultados de evaluación sobre el conjunto de test — datos del entrenamiento en Colab.")
        eval_data = _M1_EVAL[model_name]
        render_kpi_row(eval_data["metrics"])
        st.divider()
        try:
            from src.evaluation.confusion_matrix import plot_confusion_matrix
            import matplotlib.pyplot as plt
            y_true, y_pred = _cm_to_arrays(eval_data["cm"])
            fig = plot_confusion_matrix(y_true, y_pred, labels=_M1_LABELS, title=eval_data["title"])
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"No se pudo cargar el módulo de evaluación: {e}")
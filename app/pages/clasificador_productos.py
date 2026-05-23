import streamlit as st
from PIL import Image

from app.components.metrics_card import render_confidence_bars, render_kpi_row
from app.components.sidebar import model_info_card


def render() -> None:
    st.header("📦 Clasificador de Productos")
    st.caption("Clasificación automática de imágenes — EfficientNet-B0")

    model_info_card("Imágenes", "EfficientNet-B0", "Accuracy (test)", "≥ 85%")

    tab_single, tab_batch = st.tabs(["Imagen individual", "Carga por lote (CSV)"])

    with tab_single:
        uploaded = st.file_uploader("Subí una imagen del producto", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            col_img, col_result = st.columns([1, 1])
            with col_img:
                st.image(image, caption="Imagen cargada", use_column_width=True)
            with col_result:
                with st.spinner("Clasificando..."):
                    try:
                        from src.image_classifier.predict import predict
                        result = predict(image)
                        st.success(f"**{result['top_prediction']}**")
                        st.caption(f"Tiempo de inferencia: {result['inference_time_ms']} ms")
                        render_confidence_bars(result["top_3"])
                    except Exception as e:
                        st.warning(f"Modelo no disponible aún: {e}")
                        st.info("Entrenó el modelo primero corriendo el notebook 01 en Colab y copiá el .pt a models/image_classifier/")

    with tab_batch:
        st.markdown("Subí un CSV con una columna `image_path` con rutas relativas a `data/`.")
        csv_file = st.file_uploader("CSV de imágenes", type=["csv"], key="batch_csv")
        if csv_file:
            import pandas as pd
            df = pd.read_csv(csv_file)
            st.dataframe(df.head())
            if st.button("Clasificar lote"):
                st.info("Implementación pendiente — disponible en semana 5.")

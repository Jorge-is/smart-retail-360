import streamlit as st
from PIL import Image

from app.components.metrics_card import render_confidence_bars, render_kpi_row
from app.components.sidebar import model_info_card


def render() -> None:
    st.header("Clasificador de Productos")
    st.caption("Clasificación automática de imágenes — EfficientNet-B0 vs MobileNetV2")

    model_info_card("Imágenes", "EfficientNet-B0 / MobileNetV2", "Accuracy (test)", "≥ 78–85%")

    with st.sidebar:
        model_choice = st.radio(
            "Modelo activo",
            ["EfficientNet-B0 (modelo final)", "MobileNetV2 (comparación)"],
        )

    model_name = "efficientnet" if "EfficientNet" in model_choice else "mobilenetv2"

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
                        result = predict(image, model_name=model_name)
                        st.success(f"**{result['top_prediction']}**")
                        st.caption(f"Modelo: {result['model_used']} | Tiempo: {result['inference_time_ms']} ms")
                        render_confidence_bars(result["top_3"])
                    except Exception as e:
                        st.warning(f"Modelo no disponible aún: {e}")
                        st.info(
                            f"Entrenalo primero corriendo el notebook de {model_name} en Colab "
                            "y copiá el .keras a models/image_classifier/"
                        )

    with tab_batch:
        st.markdown("Subí un CSV con una columna `image_path` con rutas relativas a `data/`.")
        csv_file = st.file_uploader("CSV de imágenes", type=["csv"], key="batch_csv")
        if csv_file:
            import polars as pl
            df = pl.read_csv(csv_file)
            st.dataframe(df.head())
            if st.button("Clasificar lote"):
                st.info("Implementación pendiente — disponible en semana 5.")

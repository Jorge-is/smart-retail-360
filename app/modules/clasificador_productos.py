import streamlit as st
from PIL import Image
from components.metrics_card import render_confidence_bars, render_kpi_row
from components.sidebar import model_info_card

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
    tab_single, tab_batch = st.tabs(["Imagen individual", "Carga por lote (CSV)"])

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
                        
                        mock_result = {
                            "top_prediction": "Apparel - Shirts",
                            "model_used": "EfficientNet-B0 (Simulado)",
                            "inference_time_ms": 45,
                            "top_3": [
                                {"class": "Shirts", "confidence": 0.88},
                                {"class": "Tshirts", "confidence": 0.09},
                                {"class": "Outwear", "confidence": 0.03}
                            ]
                        }
                        st.success(f"**{mock_result['top_prediction']}**")
                        st.caption(f"Modelo: {mock_result['model_used']} | Tiempo: {mock_result['inference_time_ms']} ms")
                        render_confidence_bars(mock_result["top_3"])
                        st.session_state["images_classified"] += 1

    with tab_batch:
        st.markdown("Subi un CSV con una columna `image_path` con rutas relativas a `data/`.")
        csv_file = st.file_uploader("CSV de imágenes", type=["csv"], key="csv_productos")
        if csv_file:
            import polars as pl
            df = pl.read_csv(csv_file)
            st.dataframe(df.head())
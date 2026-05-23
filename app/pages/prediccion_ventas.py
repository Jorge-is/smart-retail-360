import streamlit as st

from app.components.charts import sales_forecast_chart
from app.components.sidebar import model_info_card


def render() -> None:
    st.header("Predicción de Ventas")
    st.caption("Pronóstico de demanda — Prophet (Meta)")

    model_info_card("Ventas", "Prophet", "MAPE", "≤ 15%")

    with st.sidebar:
        store_id = st.number_input("ID de tienda", min_value=1, max_value=1115, value=1, step=1)
        horizon = st.selectbox("Horizonte", [7, 15, 30], index=2, format_func=lambda x: f"{x} días")
        use_sentiment = st.checkbox("Incluir sentimiento como variable exógena", value=False)
        sentiment_score = None
        if use_sentiment:
            sentiment_score = st.slider("Sentimiento promedio (0=negativo, 1=positivo)", 0.0, 1.0, 0.5, 0.05)

    if st.button("Generar pronóstico"):
        with st.spinner(f"Pronosticando {horizon} días para tienda {store_id}..."):
            try:
                from src.sales_predictor.predict import predict
                result = predict(store_id=int(store_id), horizon_days=int(horizon), sentiment_score=sentiment_score)
                st.plotly_chart(sales_forecast_chart(result["forecast"]), use_container_width=True)

                import pandas as pd
                df = pd.DataFrame(result["forecast"])
                st.dataframe(df.style.format({"predicted_sales": "{:.0f}", "lower": "{:.0f}", "upper": "{:.0f}"}))

            except Exception as e:
                st.warning(f"Modelo no disponible aún: {e}")
                st.info("Entrenó Prophet corriendo el notebook 04 en Colab y copiá el .joblib a models/sales_predictor/")

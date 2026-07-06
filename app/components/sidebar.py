import streamlit as st


def model_info_card(module: str, model_name: str, metric_name: str, metric_value: str) -> None:
    """Muestra info del modelo activo en el sidebar."""
    with st.sidebar.expander(f"Modelo activo — {module}", icon=":material/model_training:"):
        st.markdown(f"**Modelo:** {model_name}")
        st.markdown(f"**{metric_name}:** {metric_value}")

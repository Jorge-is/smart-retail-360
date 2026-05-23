import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="SmartRetail 360",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.image("app/assets/logo.png") if __import__("pathlib").Path("app/assets/logo.png").exists() else st.title("🛍️ SmartRetail 360")

    selected = option_menu(
        menu_title="Módulos",
        options=["Inicio", "Clasificador de Productos", "Análisis de Sentimiento", "Predicción de Ventas", "Dashboard Integrado"],
        icons=["house", "camera", "chat-dots", "graph-up-arrow", "speedometer2"],
        default_index=0,
    )

st.sidebar.divider()
st.sidebar.caption("Proyecto Final — IA Universitaria · 2025")

if selected == "Inicio":
    st.title("SmartRetail 360")
    st.subheader("Plataforma de IA para e-commerce")

    st.markdown("""
    Bienvenido al dashboard unificado. Esta plataforma integra **tres módulos de inteligencia artificial**
    para ayudar a tiendas online a tomar decisiones críticas.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Clasificador de Productos**\nClasificación automática por imagen usando EfficientNet-B0.")
    with col2:
        st.info("**Análisis de Sentimiento**\nAnálisis de reseñas de clientes con BETO (Spanish BERT).")
    with col3:
        st.info("**Predicción de Ventas**\nPronostico de demanda con Prophet + sentimiento como variable exógena.")

    st.divider()
    st.markdown("Usá el menú lateral para navegar entre módulos.")

elif selected == "Clasificador de Productos":
    from app.pages.clasificador_productos import render
    render()

elif selected == "Análisis de Sentimiento":
    from app.pages.analisis_sentimiento import render
    render()

elif selected == "Predicción de Ventas":
    from app.pages.prediccion_ventas import render
    render()

elif selected == "Dashboard Integrado":
    from app.pages.dashboard_integrado import render
    render()

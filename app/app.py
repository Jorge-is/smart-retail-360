import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os
from pathlib import Path
from pages.eda import render as render_eda
from pages.clasificador_productos import render as render_clasificador
from pages.analisis_sentimiento import render as render_sentimiento
from pages.dashboard_integrado import render as render_dashboard
from pages.prediccion_ventas import render as render_prediccion

# Sistema de enrutamiento
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.append(app_dir)

st.set_page_config(
    page_title="SmartRetail 360",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estado global
if "images_classified" not in st.session_state:
    st.session_state["images_classified"] = 1420

if "reviews_analyzed" not in st.session_state:
    st.session_state["reviews_analyzed"] = 5280

if "avg_sentiment" not in st.session_state:
    st.session_state["avg_sentiment"] = 0.782

# Sidebar
with st.sidebar:

    logo_path = Path("app/assets/logo.png")

    if logo_path.exists():
        st.image(str(logo_path))
    else:
        st.title("SmartRetail 360")

    selected = option_menu(
    menu_title="Módulos",
    options=[
        "Inicio",
        "EDA",
        "Clasificador de Productos",
        "Análisis de Sentimiento",
        "Dashboard Integrado",
        "Predicción de Ventas",
    ],
    icons=[
        "house",
        "bar-chart",
        "camera",
        "chat-dots",
        "speedometer2",
        "graph-up-arrow",
    ],
    default_index=0,
    )

    st.divider()
    st.caption("Proyecto Final — IA Universitaria · 2026")

# Routing
if selected == "Inicio":

    st.title("SmartRetail 360")
    st.subheader("Plataforma de IA para e-commerce")

    st.markdown("""
    Bienvenido al dashboard unificado.

    Esta plataforma integra módulos de inteligencia artificial para ayudar
    a tiendas online a tomar decisiones críticas.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success(
            "**Clasificador de Productos**\n\n"
            "EfficientNet-B0 y MobileNetV2.\n\n"
            "Clasifica imágenes de productos."
        )

    with col2:
        st.success(
            "**Análisis de Sentimiento**\n\n"
            "BETO + Random Forest.\n\n"
            "Clasifica reseñas en positivo, neutro o negativo."
        )

    with col3:
        st.warning(
            "**Predicción de Ventas**\n\n"
            "Prophet (Meta).\n\n"
            "Módulo en desarrollo."
        )

    st.divider()
    st.markdown("Usa el menú lateral para navegar entre módulos.")

elif selected == "EDA":
    render_eda()

elif selected == "Clasificador de Productos":
    render_clasificador()

elif selected == "Análisis de Sentimiento":
    render_sentimiento()

elif selected == "Dashboard Integrado":
    render_dashboard()

elif selected == "Predicción de Ventas":
    render_prediccion()
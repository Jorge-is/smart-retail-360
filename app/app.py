import streamlit as st
from streamlit_option_menu import option_menu
import sys
import os
from pathlib import Path
from views.eda import render as render_eda
from views.clasificador_productos import render as render_clasificador
from views.analisis_sentimiento import render as render_sentimiento
from views.dashboard_integrado import render as render_dashboard
from views.prediccion_ventas import render as render_prediccion

# Sistema de enrutamiento
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.append(app_dir)

st.set_page_config(
    page_title="SmartRetail 360",
    page_icon=":material/storefront:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estado global — arranca en cero, se llena con uso real de los módulos
if "images_classified" not in st.session_state:
    st.session_state["images_classified"] = 0

if "reviews_analyzed" not in st.session_state:
    st.session_state["reviews_analyzed"] = 0

if "avg_sentiment" not in st.session_state:
    st.session_state["avg_sentiment"] = None

if "sentiment_counts" not in st.session_state:
    st.session_state["sentiment_counts"] = {"positive": 0, "neutral": 0, "negative": 0}

if "last_image_prediction" not in st.session_state:
    st.session_state["last_image_prediction"] = None

if "last_sentiment_prediction" not in st.session_state:
    st.session_state["last_sentiment_prediction"] = None

if "activity_log" not in st.session_state:
    st.session_state["activity_log"] = []

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
    st.caption("Proyecto Final — Inteligencia Artificial · 2026")

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
            ":material/photo_camera: **Clasificador de Productos**\n\n"
            "EfficientNet-B0 y MobileNetV2.\n\n"
            "Clasifica imágenes de productos."
        )

    with col2:
        st.success(
            ":material/chat: **Análisis de Sentimiento**\n\n"
            "BETO + Random Forest.\n\n"
            "Clasifica reseñas en positivo, neutro o negativo."
        )

    with col3:
        st.success(
            ":material/trending_up: **Predicción de Ventas**\n\n"
            "Prophet (Meta) + XGBoost.\n\n"
            "Pronostica demanda por tienda."
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
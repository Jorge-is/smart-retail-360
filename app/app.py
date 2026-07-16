import streamlit as st
import sys
import os
from pathlib import Path
from views.eda import render as render_eda
from views.clasificador_productos import render as render_clasificador
from views.analisis_sentimiento import render as render_sentimiento
from views.dashboard_integrado import render as render_dashboard
from views.prediccion_ventas import render as render_prediccion
from components.metrics_card import render_kpi_row

# Sistema de enrutamiento
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.append(app_dir)

st.set_page_config(
    page_title="SmartRetail 360",
    page_icon="app/assets/icon_smartretail.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    [data-testid="stSidebarNav"] span {
        font-size: 1.05rem;
    }
    [data-testid="stSidebarNav"] [data-testid="stIconMaterial"] {
        font-size: 1.3rem;
    }
    </style>
""", unsafe_allow_html=True)

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

if "forecasts_generated" not in st.session_state:
    st.session_state["forecasts_generated"] = 0


def render_inicio() -> None:
    st.title("SmartRetail 360")
    st.subheader("Plataforma de IA para e-commerce")

    st.markdown("""
    Bienvenido al dashboard unificado.

    Esta plataforma integra módulos de inteligencia artificial para ayudar
    a tiendas online a tomar decisiones críticas.
    """)

    render_kpi_row([
        {"label": "Imágenes clasificadas", "value": str(st.session_state["images_classified"])},
        {"label": "Reseñas analizadas", "value": str(st.session_state["reviews_analyzed"])},
        {"label": "Pronósticos generados", "value": str(st.session_state["forecasts_generated"])},
    ])
    st.caption("Contadores de esta sesión — se reinician al recargar la app.")

    st.divider()

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


with st.sidebar:
    logo_path = Path("app/assets/logo.png")

    if logo_path.exists():
        st.image(str(logo_path), width=220)
    else:
        st.title("SmartRetail 360")

pages = [
    st.Page(render_inicio, title="Inicio", icon=":material/home:", url_path="inicio", default=True),
    st.Page(render_eda, title="EDA", icon=":material/bar_chart:", url_path="eda"),
    st.Page(render_clasificador, title="Clasificador de Productos", icon=":material/photo_camera:", url_path="clasificador"),
    st.Page(render_sentimiento, title="Análisis de Sentimiento", icon=":material/chat:", url_path="sentimiento"),
    st.Page(render_dashboard, title="Dashboard Integrado", icon=":material/speed:", url_path="dashboard-integrado"),
    st.Page(render_prediccion, title="Predicción de Ventas", icon=":material/trending_up:", url_path="prediccion-ventas"),
]

pg = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.divider()
    st.caption("Proyecto Final — Inteligencia Artificial · 2026")

pg.run()

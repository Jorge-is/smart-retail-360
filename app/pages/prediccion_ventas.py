import json

import pandas as pd
import streamlit as st

from components.charts import sales_forecast_chart
from components.metrics_card import render_kpi_row
from components.sidebar import model_info_card


_RECOMMENDATION_STYLE = {
    "Apto para producción": st.success,
    "Requiere mejoras": st.warning,
    "No viable": st.error,
}


def _load_metrics() -> dict | None:
    from src.utils.config import SALES_METRICS_PATH

    if not SALES_METRICS_PATH.exists():
        return None
    with open(SALES_METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Calculando métricas de XGBoost global (primera vez)...")
def _try_auto_backfill_xgboost() -> dict | None:
    from src.utils.logging_config import get_logger

    logger = get_logger(__name__)
    try:
        from src.sales_predictor.train import backfill_xgboost_metrics
        return backfill_xgboost_metrics()
    except Exception:
        logger.exception("Falló el backfill automático de métricas XGBoost")
        # No cachear el fallo: si fue transitorio (archivo temporalmente
        # bloqueado, etc.) el próximo render() vuelve a intentarlo en vez
        # de quedar pegado en None para el resto del proceso.
        _try_auto_backfill_xgboost.clear()
        return None


def _load_historical(store_id: int) -> pd.DataFrame | None:
    from src.utils.config import DATA_DIR
    from src.sales_predictor.features import prepare_prophet_df

    raw_path = DATA_DIR / "raw" / "rossmann" / "train.csv"
    if not raw_path.exists():
        return None
    try:
        raw_df = pd.read_csv(raw_path, parse_dates=["Date"])
        df = prepare_prophet_df(raw_df, store_id=store_id)
        return df.tail(90)  # últimos ~3 meses de contexto, no todo el histórico
    except Exception:
        return None


def render() -> None:
    from src.sales_predictor.predict import predict as predict_sales
    from src.utils.config import SALES_STORE_SUBSET

    st.header("Predicción de Ventas")
    st.caption("Pronóstico de demanda por tienda — Prophet (Meta), comparado contra XGBoost")

    model_info_card("Ventas", "Prophet (Meta) — un modelo por tienda", "MAPE objetivo", "< 15%")

    with st.sidebar:
        store_id = st.selectbox(
            "Tienda",
            SALES_STORE_SUBSET,
            format_func=lambda s: f"Tienda {s}",
        )
        horizon_days = st.radio("Horizonte de pronóstico", [7, 15, 30], index=2, horizontal=True)

    tab_forecast, tab_eval = st.tabs([
        ":material/trending_up: Pronóstico",
        ":material/analytics: Comparación de modelos",
    ])

    with tab_forecast:
        st.subheader(f"Tienda {store_id} — próximos {horizon_days} días")

        avg_sentiment = st.session_state.get("avg_sentiment")

        st.markdown("##### Regressor de sentimiento")
        if avg_sentiment is not None:
            st.caption(
                f"Calculado a partir de las reseñas analizadas en esta sesión "
                f"(Análisis de Sentimiento) — {st.session_state.get('reviews_analyzed', 0)} reseñas."
            )
            manual_override = st.checkbox("Ajustar manualmente en vez de usar el valor de la sesión")
        else:
            st.caption(
                "Todavía no analizaste ninguna reseña en esta sesión — "
                "usá el slider para simular un valor, o andá a Análisis de Sentimiento primero."
            )
            manual_override = True

        if manual_override:
            sentiment_score = st.slider(
                "Sentimiento promedio (0 = muy negativo, 1 = muy positivo)",
                min_value=0.0, max_value=1.0,
                value=avg_sentiment if avg_sentiment is not None else 0.5,
                step=0.05,
            )
        else:
            sentiment_score = avg_sentiment
            st.metric("Sentimiento usado", f"{sentiment_score:.0%}")

        if st.button("Generar pronóstico", type="primary"):
            with st.spinner("Pronosticando..."):
                try:
                    result = predict_sales(
                        store_id=store_id,
                        horizon_days=horizon_days,
                        sentiment_score=sentiment_score,
                    )
                    forecast = result["forecast"]
                    historical = _load_historical(store_id)

                    if result.get("warning"):
                        st.warning(result["warning"])

                    total = sum(f["predicted_sales"] for f in forecast)
                    avg_daily = total / len(forecast)

                    render_kpi_row([
                        {"label": "Ventas totales estimadas", "value": f"${total:,.0f}"},
                        {"label": "Promedio diario", "value": f"${avg_daily:,.0f}"},
                        {"label": "Horizonte", "value": f"{horizon_days} días"},
                    ])

                    st.plotly_chart(
                        sales_forecast_chart(forecast, historical=historical),
                        use_container_width=True,
                    )
                    if historical is None:
                        st.caption(
                            "No se encontró el dataset histórico en este entorno — "
                            "se muestra solo el pronóstico, sin ventas pasadas de fondo."
                        )

                    st.session_state.setdefault("forecasts_generated", 0)
                    st.session_state["forecasts_generated"] += 1
                    st.session_state["last_forecast_store"] = store_id
                    st.session_state["activity_log"].append({
                        "modulo": "Ventas",
                        "evento": f"Pronóstico {horizon_days}d — Tienda {store_id} (sentimiento {sentiment_score:.0%})",
                        "estado": "OK",
                    })
                except FileNotFoundError as e:
                    st.error(f"Modelo no disponible: {e}")
                except Exception as e:
                    st.error(f"Error al generar el pronóstico: {e}")

    with tab_eval:
        st.caption("Resultados de evaluación sobre el set de test (últimos 30 días) — datos del entrenamiento.")
        metrics = _load_metrics()

        if metrics is None:
            st.warning(
                "No se encontró `metrics.json`. Copiá ese archivo desde el entrenamiento "
                "(`models/sales_predictor/metrics.json`) para ver esta pestaña."
            )
        else:
            st.markdown("##### Prophet — por tienda")
            prophet_rows = metrics.get("prophet_by_store", [])
            if prophet_rows:
                df_prophet = pd.DataFrame([
                    {
                        "Tienda": r["store_id"],
                        "MAE": r["prophet"]["metrics"]["mae"],
                        "RMSE": r["prophet"]["metrics"]["rmse"],
                        "MAPE (%)": r["prophet"]["metrics"]["mape"],
                        "Viabilidad": r["prophet"]["viability"]["recommendation"],
                    }
                    for r in prophet_rows
                ])
                st.dataframe(df_prophet, use_container_width=True, hide_index=True)

                selected_row = next((r for r in prophet_rows if r["store_id"] == store_id), None)
                if selected_row:
                    reco = selected_row["prophet"]["viability"]["recommendation"]
                    _RECOMMENDATION_STYLE.get(reco, st.info)(
                        f"Tienda {store_id} (seleccionada arriba): {reco}"
                    )

            st.divider()
            st.markdown("##### XGBoost — global (todas las tiendas)")
            xgb = metrics.get("xgboost_global")
            if xgb is None:
                xgb = _try_auto_backfill_xgboost()

            if xgb:
                render_kpi_row([
                    {"label": "MAE", "value": f"{xgb['xgboost']['metrics']['mae']:.2f}"},
                    {"label": "RMSE", "value": f"{xgb['xgboost']['metrics']['rmse']:.2f}"},
                    {"label": "MAPE", "value": f"{xgb['xgboost']['metrics']['mape']:.2f}%"},
                    {"label": "Tiendas cubiertas", "value": str(xgb["n_stores"])},
                ])
                reco = xgb["xgboost"]["viability"]["recommendation"]
                _RECOMMENDATION_STYLE.get(reco, st.info)(f"XGBoost global: {reco}")
                st.caption(
                    "XGBoost cubre las 1115 tiendas con un solo modelo, a costa de precisión "
                    "por tienda frente a Prophet — no se usa para pronóstico en vivo en este "
                    "dashboard, solo como punto de comparación."
                )
            else:
                st.info(
                    "No hay resultados de XGBoost global todavía — falta "
                    "`data/raw/rossmann/train.csv` y `store.csv` en el proyecto para poder "
                    "calcularlos automáticamente."
                )

import streamlit as st


def render_kpi_row(metrics: list[dict]) -> None:
    """
    metrics = [
        {
            "label": "Accuracy",
            "value": "88.5%",
            "delta": "+2.1%"
        }
    ]
    """

    cols = st.columns(len(metrics))

    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric["label"],
                value=metric["value"],
                delta=metric.get("delta")
            )


def render_confidence_bars(top3: list[dict]) -> None:
    """
    top3 = [
        {"class": "T-Shirt", "confidence": 0.91},
        {"class": "Shirt", "confidence": 0.06},
        {"class": "Dress", "confidence": 0.03},
    ]
    """

    for item in top3:

        confidence = item["confidence"]

        st.markdown(
            f"**{item['class']}** — {confidence:.2%}"
        )

        st.progress(confidence)
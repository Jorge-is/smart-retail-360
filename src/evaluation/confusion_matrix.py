"""Generación y visualización de matrices de confusión."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels: list[str],
    title: str = "Matriz de Confusión",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Genera una matriz de confusión visualizada con seaborn.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones del modelo.
        labels: Nombres de las clases.
        title: Título del gráfico.
        save_path: Si se provee, guarda la imagen en esa ruta.

    Returns:
        Figure de matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

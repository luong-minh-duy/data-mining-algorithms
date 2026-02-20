# utils/visualization.py

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from sklearn.metrics import confusion_matrix
import itertools


def plot_2d_data(
    X: np.ndarray,
    y: np.ndarray,
    title: str = "2D Data",
    xlabel: str = "Feature 1",
    ylabel: str = "Feature 2",
) -> None:
    """
    Scatter plot for 2D classification datasets.
    """

    plt.figure()
    classes = np.unique(y)

    for c in classes:
        plt.scatter(
            X[y == c, 0],
            X[y == c, 1],
            label=f"Class {c}",
            alpha=0.7,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


def plot_decision_boundary(
    model,
    X: np.ndarray,
    y: np.ndarray,
    resolution: float = 0.02,
    title: str = "Decision Boundary",
) -> None:
    """
    Plot decision boundary for 2D datasets.

    Model must implement predict().
    """

    if X.shape[1] != 2:
        raise ValueError("Decision boundary visualization requires 2D features.")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)

    classes = np.unique(y)
    for c in classes:
        plt.scatter(
            X[y == c, 0],
            X[y == c, 1],
            label=f"Class {c}",
            edgecolor="k",
        )

    plt.title(title)
    plt.legend()
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(cm))
    if class_names:
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    else:
        plt.xticks(tick_marks)
        plt.yticks(tick_marks)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def plot_probability_heatmap(
    model,
    X: np.ndarray,
    resolution: float = 0.02,
    class_index: int = 0,
    title: str = "Probability Heatmap",
) -> None:
    """
    Visualize probability surface for a specific class.

    Model must implement predict_proba().
    Only works for 2D data.
    """

    if X.shape[1] != 2:
        raise ValueError("Probability heatmap requires 2D features.")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, class_index]
    probs = probs.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, probs, alpha=0.6)
    plt.colorbar(label=f"P(class={class_index})")
    plt.title(title)
    plt.show()
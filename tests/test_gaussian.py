# tests/naive_bayes/test_gaussian.py

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.classification.naive_bayes.gaussian import GaussianNaiveBayes
from src.utils.visualization import (
    plot_decision_boundary,
    plot_confusion_matrix,
)


def test_gaussian_naive_bayes():
    # 2D dataset để visualize
    X, y = make_blobs(
        n_samples=300,
        centers=2,
        cluster_std=1.5,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n[GaussianNB] Accuracy:", round(acc, 4))
    assert acc > 0.8

    # ===== Visualization =====
    plot_decision_boundary(
        model,
        X,
        y,
        title="GaussianNB Decision Boundary",
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        title="GaussianNB Confusion Matrix",
    )
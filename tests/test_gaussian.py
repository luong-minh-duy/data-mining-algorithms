# tests/naive_bayes/test_gaussian.py

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from classification.naive_bayes.gaussian import GaussianNaiveBayes
from utils.visualization import plot_confusion_matrix


def test_gaussian_naive_bayes_iris():
    # ===== Load Iris dataset =====
    data = load_iris()
    X = data.data          # shape (150, 4)
    y = data.target        # 3 classes: 0, 1, 2

    # ===== Train / Test split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # ===== Train model =====
    model = GaussianNaiveBayes()
    model.fit(X_train, y_train)

    # ===== Prediction =====
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n[GaussianNB - Iris] Accuracy:", round(acc, 4))
    assert acc > 0.85   # Iris thường đạt > 90%

    # ===== Confusion Matrix =====
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="GaussianNB - Iris Confusion Matrix",
    )


if __name__ == "__main__":
    test_gaussian_naive_bayes_iris()
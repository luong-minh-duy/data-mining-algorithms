# naive_bayes/test_bernoulli.py

import os
import sys
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bernoulli import BernoulliNaiveBayes
from utils.visualization import plot_confusion_matrix


def test_bernoulli_naive_bayes():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    X_binary = (X > 8).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_binary, y, test_size=0.3, random_state=42
    )

    model = BernoulliNaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n[BernoulliNB] Accuracy:", round(acc, 4))
    assert acc > 0.7

    # ===== Visualization =====
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="BernoulliNB Confusion Matrix"
    )


if __name__ == "__main__":
    test_bernoulli_naive_bayes()
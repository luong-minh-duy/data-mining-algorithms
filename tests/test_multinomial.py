# tests/naive_bayes/test_multinomial.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from classification.naive_bayes.multinomial import MultinomialNaiveBayes
from utils.visualization import plot_confusion_matrix


def test_multinomial_naive_bayes():
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    X = np.clip(X, 0, None)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = MultinomialNaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\n[MultinomialNB] Accuracy:", round(acc, 4))
    assert acc > 0.7

    plot_confusion_matrix(
        y_test,
        y_pred,
        title="MultinomialNB Confusion Matrix",
    )

if __name__ == "__main__":
    test_multinomial_naive_bayes()
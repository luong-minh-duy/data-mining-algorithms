# naive_bayes/multinomial.py

from __future__ import annotations
import numpy as np
from typing import Optional


class MultinomialNaiveBayes:
    """
    Multinomial Naive Bayes classifier.

    Suitable for count data (e.g., text classification).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.classes_: Optional[np.ndarray] = None
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_, counts = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape

        self.class_log_prior_ = np.log(counts / n_samples)
        self.feature_log_prob_ = np.zeros((len(self.classes_), n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_count = X_c.sum(axis=0)
            smoothed = feature_count + self.alpha
            smoothed_total = smoothed.sum()
            self.feature_log_prob_[idx, :] = np.log(smoothed / smoothed_total)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        log_probs = X @ self.feature_log_prob_.T + self.class_log_prior_
        log_probs -= log_probs.max(axis=1, keepdims=True)

        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
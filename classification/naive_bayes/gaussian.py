# naive_bayes/gaussian.py

from __future__ import annotations
import numpy as np
from typing import Optional


class GaussianNaiveBayes:
    """
    Gaussian Naive Bayes classifier.

    Assumes each feature follows a normal distribution:
        P(x_i | C) ~ N(mu_C, var_C)

    Uses log-probabilities for numerical stability.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: Optional[np.ndarray] = None
        self.class_prior_: Optional[np.ndarray] = None
        self.theta_: Optional[np.ndarray] = None  # means
        self.var_: Optional[np.ndarray] = None    # variances

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_, counts = np.unique(y, return_counts=True)
        n_samples, n_features = X.shape

        self.class_prior_ = counts / n_samples
        self.theta_ = np.zeros((len(self.classes_), n_features))
        self.var_ = np.zeros((len(self.classes_), n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0) + self.var_smoothing

        return self

    def _log_gaussian_pdf(self, class_idx: int, X: np.ndarray) -> np.ndarray:
        mean = self.theta_[class_idx]
        var = self.var_[class_idx]

        return -0.5 * (
            np.sum(np.log(2.0 * np.pi * var))
            + np.sum(((X - mean) ** 2) / var, axis=1)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        log_probs = []

        for idx, _ in enumerate(self.classes_):
            log_prior = np.log(self.class_prior_[idx])
            log_likelihood = self._log_gaussian_pdf(idx, X)
            log_probs.append(log_prior + log_likelihood)

        log_probs = np.vstack(log_probs).T
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
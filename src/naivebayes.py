import numpy as np
import pandas as pd
from typing import Union, Optional

class ScratchNaiveBayes:
    def __init__(self, epsilon: float = 1e-9):
        self.epsilon = epsilon
        self.classes: Optional[np.ndarray] = None
        self.class_probs: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.vars: Optional[np.ndarray] = None  # Store variance instead of std for efficiency
        self._fitted = False

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]):
        """Validate and preprocess input data."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X.shape[0] != len(y):
            raise ValueError(f"Found {X.shape[0]} samples in X but {len(y)} in y")
        if not np.isfinite(X).all():
            raise ValueError("X contains non-finite values")
        
        if isinstance(y, pd.DataFrame):
            y = np.array(y).ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if not np.isfinite(y).all():
            raise ValueError("y contains non-finite values")
        return X, y

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]):
        """
        Fit the Gaussian Naive Bayes model using vectorized operations.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            self: Returns the instance itself
        """
        X, y = self._validate_input(X, y)
        
        # Compute class-related statistics
        self.classes = np.unique(y)
        n_samples = X.shape[0]
        
        # Vectorized computation of class probabilities
        class_counts = np.bincount(y, minlength=len(self.classes))
        self.class_probs = class_counts / n_samples
        
        # Pre-allocate arrays for means and variances
        n_features = X.shape[1]
        self.means = np.zeros((len(self.classes), n_features))
        self.vars = np.zeros((len(self.classes), n_features))
        
        # Vectorized computation of means and variances
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i] = np.mean(X_c, axis=0)
            self.vars[i] = np.var(X_c, axis=0) + self.epsilon
        
        self._fitted = True
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Predict class probabilities using vectorized operations.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Predicted class probabilities of shape (n_samples, n_classes)
        """
        if not self._fitted:
            raise RuntimeError("The model must be fitted before making predictions")
            
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
            
        # Compute log probabilities for all classes at once
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, len(self.classes)))
        
        # Vectorized computation of log probabilities
        for i, _ in enumerate(self.classes):
            # Compute log likelihood using broadcasting
            diff = X - self.means[i]
            log_likelihood = -0.5 * (
                np.sum(np.log(2 * np.pi * self.vars[i]))
                + np.sum((diff ** 2) / self.vars[i], axis=1)
            )
            log_probs[:, i] = np.log(self.class_probs[i]) + log_likelihood
        
        # Numerical stability using log-sum-exp trick
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_probs = np.exp(log_probs - max_log_probs)
        probs = exp_probs / (np.sum(exp_probs, axis=1, keepdims=True) + self.epsilon)
        
        return probs

    def predict(self, X: Union[np.ndarray, pd.DataFrame]):
        """
        Predict class labels using vectorized operations.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.DataFrame]):
        """
        Compute accuracy score.
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Accuracy score
        """
        return np.mean(self.predict(X) == y)
import numpy as np
from sklearn.ensemble import IsolationForest
from .base import BaseDetector

class IsolationForestDetector(BaseDetector):
    """
    Wrapper for the scikit-learn IsolationForest model.

    This model isolates observations by randomly selecting a feature and then
    randomly selecting a split value between the maximum and minimum values of
    the selected feature. The number of splittings required to isolate a sample
    is equivalent to the path length in a tree. Anomalies are expected to have
    shorter average path lengths.
    """
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=42):
        """
        Args:
            n_estimators (int): The number of base estimators (trees) in the ensemble.
            max_samples (int or float): The number of samples to draw from X to train each base estimator.
            contamination (float or 'auto'): The amount of contamination of the data set. We ignore this.
            random_state (int): Seed for the random number generator for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        
        # Instantiate the underlying sklearn model
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state
        )

    def fit(self, X, y=None):
        """
        Fit the Isolation Forest model.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
        """
        self.model_.fit(X)
        return self

    def decision_function(self, X):
        """
        Calculate the anomaly score for each sample.
        Scikit-learn's IsolationForest score is inverted (lower is more anomalous).
        We will flip the sign so that higher scores indicate anomalies, conforming
        to our convention.

        Args:
            X (np.ndarray): Data to score, shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Anomaly scores, shape (n_samples,).
        """
        # The `score_samples` method returns the opposite of the anomaly score.
        # We negate it to conform to the convention (higher score = more anomalous).
        return -self.model_.score_samples(X)
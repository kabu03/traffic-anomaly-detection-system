import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .base import BaseDetector

class LOFDetector(BaseDetector):
    """
    Wrapper for the scikit-learn LocalOutlierFactor model.

    LOF measures the local deviation of density of a given sample with respect
    to its neighbors. It is local in that the anomaly score depends on how
    isolated the object is with respect to the surrounding neighborhood.
    """
    def __init__(self, n_neighbors=20, contamination='auto'):
        """
        Args:
            n_neighbors (int): Number of neighbors to use for local density estimation.
                               This is the most important hyperparameter.
            contamination (float or 'auto'): The amount of contamination of the data set.
                                             We ignore this and use our own thresholding.
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        # LOF in 'novelty' detection mode requires fitting on normal data first.
        # We set novelty=True to enable the decision_function method on new data.
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True
        )

    def fit(self, X, y=None):
        """
        Fit the LOF model to the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
        """
        self.model_.fit(X)
        return self

    def decision_function(self, X):
        """
        Calculate the anomaly score for each sample.
        Scikit-learn's LOF score is inverted (lower is more anomalous).
        We will flip the sign so that higher scores indicate anomalies.

        Args:
            X (np.ndarray): Data to score, shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Anomaly scores, shape (n_samples,).
        """
        # The `score_samples` method returns the opposite of the anomaly score.
        # We negate it to conform to the convention (higher score = more anomalous).
        return -self.model_.score_samples(X)
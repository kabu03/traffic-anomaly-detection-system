import numpy as np
from sklearn.svm import OneClassSVM
from .base import BaseDetector

class OneClassSVMDetector(BaseDetector):
    """
    Wrapper for the scikit-learn OneClassSVM model.

    This model learns a decision boundary that encompasses the majority of the
    training data. New points are scored based on their signed distance to this
    boundary.
    """
    def __init__(self, kernel='rbf', nu=0.05, gamma='scale'):
        """
        Args:
            kernel (str): The kernel type to be used in the algorithm. 'rbf' is standard.
            nu (float): An upper bound on the fraction of training errors and a lower
                        bound of the fraction of support vectors. Key hyperparameter.
            gamma (str or float): Kernel coefficient for 'rbf'. 'scale' is a good default.
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma

        self.model_ = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )

    def fit(self, X, y=None):
        """
        Fit the One-Class SVM model.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
        """
        self.model_.fit(X)
        return self

    def decision_function(self, X):
        """
        Calculate the anomaly score for each sample.
        Scikit-learn's OneClassSVM `decision_function` returns higher values for
        inliers. We negate it to conform to our convention (higher score = more anomalous).

        Args:
            X (np.ndarray): Data to score, shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Anomaly scores, shape (n_samples,).
        """
        # Negate the score to conform to our convention
        return -self.model_.decision_function(X)
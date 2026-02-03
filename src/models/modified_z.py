import numpy as np
from .base import BaseDetector

class ModZScoreDetector(BaseDetector):
    """
    Anomaly detector using the Modified Z-Score method.

    This method is robust to outliers as it uses the median and
    Median Absolute Deviation (MAD) instead of mean and standard deviation.
    The anomaly score is the absolute value of the modified z-score.
    """
    def __init__(self):
        # This simple model has no hyperparameters to tune.
        pass

    def fit(self, X, y=None):
        """
        Calculate and store the median and MAD from the training data.

        Args:
            X (np.ndarray): Training data of shape (n_samples, n_features).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        self.median_ = np.median(X, axis=0)
        
        # Calculate Median Absolute Deviation (MAD)
        diff = np.abs(X - self.median_)
        self.mad_ = np.median(diff, axis=0)
        
        return self

    def decision_function(self, X):
        """
        Calculate the anomaly score for each sample using the modified z-score.

        Args:
            X (np.ndarray): Data to score, shape (n_samples, n_features).
        
        Returns:
            np.ndarray: Anomaly scores, shape (n_samples,).
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Create a safe version of MAD to avoid division by zero. If MAD is 0,
        # it means all training data points were identical. Any deviation is
        # technically infinite, but we'll handle it to avoid errors.
        # We replace mad=0 with 1; the corresponding diff will also be 0, so score is 0.
        mad_safe = np.where(self.mad_ == 0, 1.0, self.mad_)
        
        diff = np.abs(X - self.median_)
        
        # Calculate modified z-scores for each feature
        mod_z_scores = 0.6745 * diff / mad_safe
        
        # For multi-feature input, a common approach is to use the max score
        # across features for each sample. This means a sample is as anomalous
        # as its most anomalous feature.
        if X.shape[1] > 1:
            return np.max(mod_z_scores, axis=1)
        else:
            return mod_z_scores.flatten()
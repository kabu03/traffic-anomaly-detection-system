import abc
from sklearn.base import BaseEstimator, OutlierMixin

class BaseDetector(BaseEstimator, OutlierMixin, abc.ABC):
    """
    Abstract base class for all anomaly detection models.

    This ensures a consistent API (fit, decision_function) compatible with
    scikit-learn, which will be crucial for the pipelines.
    """

    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Fit the detector to the training data. For unsupervised models,
        y is typically ignored.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abc.abstractmethod
    def decision_function(self, X):
        """
        Calculate the raw anomaly score for each sample in X.
        By convention, higher scores indicate a higher likelihood of being an anomaly.

        This method must be implemented by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
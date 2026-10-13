from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score

from models import BaseNMF

class Evaluation:
    """Evaluation metrics (as per assignment1.ipynb) and routines."""

    def __init__(
            self, 
            V : np.ndarray,
            Y : np.ndarray,
            seed : int = 42
            ) -> None:
        
        self.V = V
        self.Y = Y
        self.seed = seed

    def relative_reconstruction_error(self, reconstruction):
        """RRE = || V - W H ||_F / || V ||_F"""

        rre = np.linalg.norm(self.V - reconstruction) / np.linalg.norm(self.V)
        return rre

    def assign_label(self, transformed):
        """Perform K means clustering followed by majority voting based on true labels"""
        
        clusters = KMeans(n_clusters=len(set(self.Y)), random_state=self.seed).fit(transformed.T)
        Y_pred = np.zeros(self.Y.shape)
        for i in set(clusters.labels_):
            ind = clusters.labels_ == i
            Y_pred[ind] = Counter(self.Y[ind]).most_common(1)[0][0]
        return Y_pred

    def average_accuracy(self, pred):
        """Acc(Y, pred) = (sum_n 1{pred_i = Y_i}) / n"""

        acc = accuracy_score(self.Y, pred)
        return acc

    def normalized_mutual_information(self, pred):
        """NMI(Y, pred) = 2 * I(Y, pred) / (H(Y) + H(pred))"""

        nmi = normalized_mutual_info_score(self.Y, pred)
        return nmi

    def evaluate(self, model : BaseNMF):
        """
        Takes a trained model and performs evaluations on V. Returns the scores for
        relative reconstruction error, average accuracy and normalized mutual information.
        """

        dictionary, transformed = model.transform()
        reconstruction = model.reconstruct()
        pred = self.assign_label(transformed)

        rre = self.relative_reconstruction_error(reconstruction)
        acc = self.average_accuracy(pred)
        nmi = self.normalized_mutual_information(pred)
        return rre, acc, nmi

# Set of evaluation metrics that are available, to be used by visualization.py
METRICS = ('rre', 'acc', 'nmi')


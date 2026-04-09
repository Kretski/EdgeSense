import numpy as np
from scipy.stats import ks_2samp

class DriftDetector:
    """Откриване на drift в нови данни спрямо референтни"""

    def __init__(self, ref_data):
        self.ref_data = ref_data
        self.ref_mean = np.mean(ref_data, axis=0)

    def detect(self, new_data):
        # KS тест на всяка колона
        drift_scores = []
        for i in range(new_data.shape[1]):
            stat, p = ks_2samp(self.ref_data[:, i], new_data[:, i])
            drift_scores.append(p < 0.05)
        return any(drift_scores)

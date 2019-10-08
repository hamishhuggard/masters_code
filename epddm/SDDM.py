import numpy as np

class SDDM: # Surprisal drift detection method

    def __init__(self, drift_threshold, warning_threshold, n_samples=1000, two_tailed=False):
        self.n_samples = n_samples
        self.drift_theshold = drift_threshold
        self.warning_threshold = warning_threshold
        # If two_tailed then test for cumulative surprisal error is either too
        # high or two low. The former indicates that noise has decreased, the
        # latter that noise has increased or that the decision boundary has shifted.
        # In most cases we only care about the latter.
        self.two_tailed = two_tailed
        if two_tailed: # Bonferroni correction
            self.drive_threshold /= 2
            self.warning_threshold /= 2
        self.monte_carlo = [ 0 for in range(n_samples) ]
        self.cusum = 0

    def update(self, probs, true_label):

        surprisals = -np.log(probs)
        surprisal = -np.log(probs[true_label])
        expected_surprisal = np.sum(-probs * np.log(probs))
        surprisal_error = surprisal - expected_surprisal
        self.cusum += surprisal_error

        # Update monte carlo
        self.monte_carlo += np.random.choice(-np.log(probs),
                                size=self.n_sampels, replace=True, p=probs)

        # Calculate improbability
        if self.two_tailed:
            pr_lower = np.sum([i <= self.cusum for i in range(self.n_samples)])
            pr_lower /= self.n_samples
            if pr_lower < warning_threshold:
                self.warning = True
            if pr_lower < drift_threshold:
                self.drift = True

        pr_upper = np.sum([i >= self.cusum for i in range(self.n_samples)])
        pr_upper /= self.n_samples
        if pr_upper < warning_threshold:
            self.warning = True
        if pr_upper < drift_threshold:
            self.drift = True

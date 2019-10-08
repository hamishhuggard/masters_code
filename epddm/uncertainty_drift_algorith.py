# drift detection with uncertainty
import numpy as np

class EpDDM: # Epistemic Drift Detection Method
    def __init__(self, drift_threshold=0.01,
                 buffer_size=100, tail_size=10):

        # Parameters
        self.drift_threshold = drift_threshold
        self.drift_threshold /= (buffer_size + tail_size) # Bonferroni correction
        self.buffer_size = buffer_size
        self.tail_size = tail_size

        # Data storage
        self.buffer_x = [ None for i in range(buffer_size) ]
        self.buffer_y = [ None for i in range(buffer_size) ]
        self.means = [ None for i in range(tail_size + buffer_size) ]
        self.means_n = [ 0 for i in range(tail_size + buffer_size) ]
        self.drop_next = [ True for i in range(tail_size) ]

        # Calcalations
        self.hoeffding_bounds = []
        self.warning = False
        self.drift = False

    def reset_buffer(self):
        self.buffer = [ None for i in range(len(buffer_size)) ]

    def update(self, probability_distribution, true_label, x=None):

        # Current observation
        expected_p = np.sum(probability_distribution**2)
        observed_p = probability_distribution[true_label]
        p_error = observed_p - expected_p

        # Update means values
        self.means *= self.means_n / (self.means_n + 1)
        self.means += p_error / (self.means_n + 1)
        self.means_n += 1
        self.means = np.concatenate(([p_error], self.means))
        self.means_n = np.concatenate(([1], self.means_n))

        # Update buffer values
        self.buffer_x = np.concatenate(([x], self.buffer_x[:-1]))
        self.buffer_y = np.concatenate(([true_label], self.buffer_y[:-1]))

        # Delete one of the mean values (on the tail)
        if not any(self.drop_next):
            # If all tail positions have already been dropped then reset
            self.drop_next = [ True for i in range(self.tail_n) ]
        # Find the index for the next tail position to be dropped
        i = 0
        while not self.drop_next[i]:
            self.drop_next[i] = True
            i += 1
        self.dropped[i] = False
        # Remove entry i from the tail
        self.means = np.delete(self.means, self.buffer_size + i)
        self.means_n = np.delete(self.means_n, self.buffer_size + i)

        # Test if any of the means' improbabilities have dropped below a threshold
        # The Hoeffding bound is Pr(|X_| >= t) <= 2 exp( - n t^2 / 2)
        # where X_ is the mean of sequence, and n is the length of the sequence
        self.hoeffding_bounds = 2 * np.exp( - self.means_n * self.means**2 / 2 )

    def needs_retrain(self):

        # Get the index of the lowest-probability mean.
        # In the case of draws, go with the older one so that
        # we have more training data.
        min_i = 0
        min_prob = np.inf
        for i, prob in enumerate(self.hoeffding_bounds):
            if prob <= min_prob:
                min_prob = prob
                min_i = i

        # If none of the means are sufficiently improbable
        if min_prob > self.drift_threshold:
            return False

        # Otherwise return the data needed for retraining
        if min_i >= self.buffer_size:
            return self.buffer
        else:
            return self.buffer[:min_i]

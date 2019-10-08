import numpy as np

class EpDDM: # Epistemic Drift Detection Method, or Expected Probability Drift Detection Method
    def __init__(self, drift_threshold=0.01, buffer_size=100, tail_size=10):

        # Parameters
        self.drift_threshold = drift_threshold
        self.drift_threshold /= (buffer_size + tail_size) # Bonferroni correction
        self.buffer_size = buffer_size
        self.tail_size = tail_size

        # Data storage
        self.buffer_x = [ None for i in range(buffer_size) ]
        self.buffer_y = [ None for i in range(buffer_size) ]
        self.means = np.array([ 0 for i in range(tail_size + buffer_size) ], dtype='float64')
        self.means_n = np.array([ 0 for i in range(tail_size + buffer_size) ], dtype='float64')
        self.exp_denominators = np.array([ 0 for i in range(tail_size + buffer_size) ], dtype='float64')
        self.drop_next = [ True for i in range(tail_size) ]

        # Calcalations
        self.hoeffding_bounds = []
        self.warning = False
        self.drift = False
        self.n_samples = 0

    def update(self, probability_distribution, true_label, x=None):
        
        self.n_samples += 1

        # Current observation
        expected_p = np.sum(probability_distribution**2)
        possible_p_errors = probability_distribution - expected_p
        p_error_range = np.max(possible_p_errors) - np.min(possible_p_errors)
        observed_p = probability_distribution[true_label]
        p_error = observed_p - expected_p
        
        #print(probability_distribution, true_label, x)
        #for thing in 'expected_p possible_p_errors p_error_range observed_p p_error'.split():
        #    print(thing, eval(thing))

        # Update means values
        self.means *= self.means_n / (self.means_n + 1)
        self.means += p_error / (self.means_n + 1)
        self.means_n += 1
        self.means = np.concatenate(([p_error], self.means))
        self.means_n = np.concatenate(([1], self.means_n))

        # Update buffer values
        #print([x], self.buffer_x[:-1])
        self.buffer_x = [x] + self.buffer_x[:-1]
        self.buffer_y = [true_label] + self.buffer_y[:-1]

        # Update exp denominators
        self.exp_denominators =  np.concatenate(([ 0 ], self.exp_denominators))
        self.exp_denominators += p_error_range**2

        # Delete one of the mean values (on the tail)
        if not any(self.drop_next):
            # If all tail positions have already been dropped then reset
            self.drop_next = [ True for i in range(self.tail_size) ]
        # Find the index for the next tail position to be dropped
        i = 0
        while not self.drop_next[i]:
            self.drop_next[i] = True
            i += 1
        self.drop_next[i] = False
        # Remove entry i from the tail
        self.means = np.delete(self.means, self.buffer_size + i)
        self.means_n = np.delete(self.means_n, self.buffer_size + i)
        self.exp_denominators = np.delete(self.exp_denominators, self.buffer_size + i)

        # Hoeffding bound: Pr(|sum_i X_i / n| >= t) <= 2 exp( - 2 n^2 t^2 / sum_i (b_i-a_i)^2 )
        self.hoeffding_bounds = 2 * np.exp( - 2 * self.means_n**2 * self.means**2 \
                                    / self.exp_denominators )
        
#         for thing in 'means means_n exp_denominators hoeffding_bounds drop_next'.split():
#             print(thing)
#             print(eval('self.'+thing))

    def needs_retrain(self):

        # Get the index of the lowest-probability mean.
        # In the case of draws, go with the older one so that
        # we have more training data.
        min_i = 0
        min_prob = np.inf
        for i, prob in enumerate(self.hoeffding_bounds[:self.n_samples]):
            if prob <= min_prob:
                min_prob = prob
                min_i = i

        # If none of the means are sufficiently improbable
        if min_prob > self.drift_threshold:
            return None

        # Otherwise return the data needed for retraining
        if min_i >= self.buffer_size:
            return self.buffer_x, self.buffer_y
        else:
            return self.buffer_x[:min_i], self.buffer_y[:min_i]

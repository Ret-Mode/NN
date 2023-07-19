import math
import random

class RBM:
    def __init__(self, visible_size, hidden_size):
        self.visible_size = visible_size
        self.hidden_size = hidden_size

        self.visible_units = [0] * visible_size
        self.hidden_units = [0] * hidden_size

        self.weights = [[random.random() for _ in range(hidden_size)] for _ in range(visible_size)]
        self.visible_bias = [random.random() for _ in range(visible_size)]
        self.hidden_bias = [random.random() for _ in range(hidden_size)]

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def _sample_binary(self, prob):
        return 1 if random.random() < prob else 0

    def _energy(self, visible_units, hidden_units):
        energy = 0.0

        for i in range(self.visible_size):
            for j in range(self.hidden_size):
                energy -= self.weights[i][j] * visible_units[i] * hidden_units[j]

        for i in range(self.visible_size):
            energy -= self.visible_bias[i] * visible_units[i]

        for j in range(self.hidden_size):
            energy -= self.hidden_bias[j] * hidden_units[j]

        return energy

    def _visible_prob(self, hidden_units):
        visible_probs = [self._sigmoid(sum(self.weights[i][j] * hidden_units[j] for j in range(self.hidden_size)) + self.visible_bias[i]) for i in range(self.visible_size)]
        return visible_probs

    def _hidden_prob(self, visible_units):
        hidden_probs = [self._sigmoid(sum(self.weights[i][j] * visible_units[i] for i in range(self.visible_size)) + self.hidden_bias[j]) for j in range(self.hidden_size)]
        return hidden_probs

    def train(self, input_data, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            for input_vector in input_data:
                # Gibbs sampling for contrastive divergence
                visible_probs_0 = list(input_vector)
                hidden_probs_0 = self._hidden_prob(visible_probs_0)
                hidden_states_0 = [self._sample_binary(prob) for prob in hidden_probs_0]

                for _ in range(10):  # Run for a few iterations (adjust as needed)
                    visible_probs_k = self._visible_prob(hidden_states_0)
                    visible_states_k = [self._sample_binary(prob) for prob in visible_probs_k]
                    hidden_probs_k = self._hidden_prob(visible_states_k)
                    hidden_states_0 = [self._sample_binary(prob) for prob in hidden_probs_k]

                # Update weights and biases
                for i in range(self.visible_size):
                    for j in range(self.hidden_size):
                        self.weights[i][j] += learning_rate * (visible_probs_0[i] * hidden_probs_0[j] - visible_probs_k[i] * hidden_probs_k[j])

                for i in range(self.visible_size):
                    self.visible_bias[i] += learning_rate * (visible_probs_0[i] - visible_probs_k[i])

                for j in range(self.hidden_size):
                    self.hidden_bias[j] += learning_rate * (hidden_probs_0[j] - hidden_probs_k[j])



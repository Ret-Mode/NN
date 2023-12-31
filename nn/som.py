# autogenerated


import math
import random

class SOM:
    def __init__(self, input_size, map_size):
        self.input_size = input_size
        self.map_size = map_size
        self.weights = [[[random.random() for _ in range(input_size)] for _ in range(map_size[1])] for _ in range(map_size[0])]

    def _find_best_matching_unit(self, input_vector):
        min_dist = float('inf')
        best_unit = (0, 0)

        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(input_vector, self.weights[i][j])))
                if dist < min_dist:
                    min_dist = dist
                    best_unit = (i, j)

        return best_unit

    def train(self, input_data, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            for input_vector in input_data:
                bmu = self._find_best_matching_unit(input_vector)

                # Update the weights of the best matching unit and its neighbors
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        dist_to_bmu = math.sqrt((i - bmu[0]) ** 2 + (j - bmu[1]) ** 2)
                        influence = math.exp(-dist_to_bmu / 2)
                        for k in range(self.input_size):
                            self.weights[i][j][k] += learning_rate * influence * (input_vector[k] - self.weights[i][j][k])


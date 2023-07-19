from random import random, seed
seed(0)

# Perceptron test
from nn.perceptron import Perceptron

perceptron = Perceptron(input_size=2)

print(perceptron.predict([0.6, 0.7]))
for i in range(10000):
    x1 = random()
    x2 = random()
    truth = float(x1 * x2 > 0.4)
    perceptron.train([x1, x2], truth)
print(perceptron.predict([0.6, 0.7]))


# MLP test
from nn.mlp import MLP

mlp = MLP(input_size=2, hidden_size=2, output_size=1)

print(mlp.forward([0.6, 0.7]))
for i in range(10000):
    x1 = random()
    x2 = random()
    truth = float(x1 * x2 > 0.4)
    mlp.train([x1, x2], [truth])
print(mlp.forward([0.6, 0.7]))


# RBFN test
from nn.rbfn import RBFN

rbfn = RBFN(input_size=2, hidden_size=3, output_size=1)

print(rbfn.forward([0.6, 0.7]))
for i in range(10000):
    x1 = random()
    x2 = random()
    truth = float(x1 * x2 > 0.4)
    rbfn.train([x1, x2], truth)
print(rbfn.forward([0.6, 0.7]))


# SOM test
from nn.som import SOM

som = SOM(input_size=2, map_size=(5, 5))

print(som.weights)
data = [[random(), random()] for _ in range(1000)]
som.train(data, learning_rate=0.1, epochs=100)
print(som.weights)


# Hopfield Network test
from nn.hopfield import HopfieldNetwork

hopfield_net = HopfieldNetwork(num_neurons=4)

print(hopfield_net.predict([1, -1, 1, -1]))
patterns = [[1, 1, -1, -1], [-1, -1, 1, 1], [1, -1, 1, -1]]
hopfield_net.train(patterns)
print(hopfield_net.predict([1, -1, 1, -1]))


# Restricted Boltzmann Machine test
from nn.rbm import RBM

rbm = RBM(visible_size=4, hidden_size=2)

print(rbm.visible_units)
data = [[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
rbm.train(data, learning_rate=0.1, epochs=100)
print(rbm.visible_units)


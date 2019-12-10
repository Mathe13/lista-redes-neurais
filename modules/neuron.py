import numpy as np
from modules.math_functions import math_functions


class Neuron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01, activation=math_functions.binary):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
        self.activation = activation

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * \
                    (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                self.error = (label - prediction)

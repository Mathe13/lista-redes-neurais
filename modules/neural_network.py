from numpy import exp, array, random, dot, append
from modules.math_functions import math_functions
import numpy as np


class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):

        self.synaptic_weights = (2 * random.random(
            (number_of_inputs_per_neuron, number_of_neurons)) - 1)


class NeuralNetwork():
    def __init__(self, layers, activation=math_functions.sigmoid, delta=math_functions.dSigmoid, learning_rate=1):
        self.layers = layers
        self.__activation = activation
        self.__delta_funcion = delta
        self.learning_rate = learning_rate

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for __ in range(number_of_training_iterations):
            outputs = self.think(
                training_set_inputs)
            layers_error = [0] * len(self.layers)
            layers_delta = [0] * len(self.layers)
            for i in range(len(self.layers)-1, -1, -1):
                if(i == len(self.layers)-1):
                    layers_error[i] = training_set_outputs - outputs[i]
                    layers_delta[i] = layers_error[i] * \
                        self.__delta_funcion(outputs[i])

                else:
                    layers_error[i] = layers_delta[(i+1)].dot(
                        self.layers[(i+1)].synaptic_weights.T)
                    layers_delta[i] = layers_error[i] * \
                        self.__delta_funcion(outputs[i])

            for i in range(len(self.layers)-1, -1, -1):
                if(i == 0):
                    self.layers[i].synaptic_weights += (self.learning_rate*(training_set_inputs.T.dot(
                        layers_delta[i])))
                else:
                    self.layers[i].synaptic_weights += (self.learning_rate*outputs[i-1].T.dot(
                        layers_delta[i]))

    def think(self, inputs):
        outputs = []
        for i in range(len(self.layers)):
            if(i == 0):
                outputs.append(self.__activation(
                    dot(inputs, self.layers[i].synaptic_weights)))
            else:
                outputs.append(self.__activation(
                    dot(outputs[i-1], self.layers[i].synaptic_weights)))
        return outputs

    def print_weights(self):
        for i in range(len(self.layers)):
            print("Layer ", i, " : ")
            print(self.layers[i].synaptic_weights)

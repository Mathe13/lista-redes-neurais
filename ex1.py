import numpy as np
from modules.neuron import Neuron
from modules.math_functions import math_functions


# and
print("criando neuronio and")
training_inputsAnd = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
labelsAnd = np.array([1, 0, 0, 0])
neuronAnd = Neuron(2, activation=math_functions.binary)
neuronAnd.train(training_inputsAnd, labelsAnd)
print("1 and 1 = ", neuronAnd.predict(np.array([1, 1])))
print("1 and 0 = ", neuronAnd.predict(np.array([0, 1])))


# or
print("criando neuronio or")

training_inputsOr = np.array([[1, 1], [1, 0], [0, 1], [0, 0], ])
labelsOr = np.array([1, 1, 1, 0])
neuronOr = Neuron(2)
neuronOr.train(training_inputsOr, labelsOr)
print("1 or 1 = ", neuronOr.predict(np.array([1, 1])))
print("1 or 0 = ", neuronOr.predict(np.array([0, 1])))

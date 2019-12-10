
from modules.math_functions import math_functions
from numpy import array, random
from modules.neural_network import NeuralNetwork, NeuronLayer
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import show
import numpy as np
from modules.neural_network import NeuronLayer, NeuralNetwork
import matplotlib.pyplot as plt

# ------------- INPUT GENERATION -------------

# SINE WAVE DESIRED

fig = plt.figure()

plt.subplot(4, 1, 1)

time = np.arange(0, 10, 0.1)

signal = np.sin(time)
print(len(signal))
plt.title('Sine wave desired')

plt.ylabel('Signal')

plt.plot(time, signal)

# SINE WAVE WITH NOISE

plt.subplot(4, 1, 2)

noise = 0.3 * np.random.normal(0, 1, 100)

signalNoise = signal + noise

plt.plot(time, signalNoise)

plt.title('Sine wave with noise')

plt.ylabel('Signal')

# SINE WAVE FILTERED BY RC FILTER

K = 0.01
# y(n) = K*x(n-1) + (1-K)*y(n-1)

signalFilteredRC = []

dataset_input = []
dataset_output = []

for i, value in enumerate(time):

    signalFilteredRC.append(K * time[i - 1] + (1 - K) * signalNoise[i - 1])

    dataset_input.append([time[i], signalFilteredRC[i]])
    dataset_output.append([signal[i]])

plt.subplot(4, 1, 3)

plt.plot(time, signalFilteredRC)

plt.title('Sine wave filtered by RC filter')

plt.xlabel('Time (s)')
plt.ylabel('Signal')


# rede

# noises
noise = 5 * np.random.normal(0, 1, 100)
signalNoise1 = signal + noise

noise = 0.5 * np.random.normal(0, 1, 100)
signalNoise2 = signal + noise

noise = 3 * np.random.normal(0, 1, 100)
signalNoise3 = signal + noise

noise = -2 * np.random.normal(0, 1, 100)
signalNoise4 = signal + noise

noise = 0.2 * np.random.normal(0, 1, 100)
signalNoise5 = signal + noise

noise = 1 * np.random.normal(0, 1, 100)
signalNoise6 = signal + noise

noise = -2.5 * np.random.normal(0, 1, 100)
signalNoise7 = signal + noise

noise = -1 * np.random.normal(0, 1, 100)
signalNoise8 = signal + noise

noise = 0.25 * np.random.normal(0, 1, 100)
signalNoise9 = signal + noise

random.seed(16)
layer1 = NeuronLayer(200, 100)
layer2 = NeuronLayer(100, 200)
# print(signalNoise1)
neural_network = NeuralNetwork(
    [layer1, layer2], learning_rate=0.5)
training_set_inputs = array(
    [signalNoise1, signalNoise2, signalNoise3, signalNoise4, signalNoise5, signalNoise6, signalNoise7, signalNoise8, signalNoise9])
# print(np.shape(training_set_inputs))
training_set_outputs = array(
    [signal, signal, signal, signal, signal, signal, signal, signal, signal])
# print(np.shape(training_set_outputs))

neural_network.train(training_set_inputs, training_set_outputs, 40000)
outputs = neural_network.think(signalNoise)
plt.subplot(4, 1, 4)
plt.plot(time, outputs[len(outputs)-1])

plt.title('Sine wave after AI')

plt.ylabel('Signal')

show()

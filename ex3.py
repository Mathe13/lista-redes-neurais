from modules.neural_network import NeuralNetwork, NeuronLayer
from numpy import array, random
from modules.math_functions import math_functions

if __name__ == "__main__":
    random.seed(16)
    layer1 = NeuronLayer(4, 2)
    layer2 = NeuronLayer(3, 4)
    layer3 = NeuronLayer(1, 3)
    neural_network = NeuralNetwork(
        [layer1, layer2, layer3], activation=math_functions.relu, delta=math_functions.dRelu, learning_rate=0.0001)
    print("Iniciando com pesos randomicos")
    neural_network.print_weights()
    training_set_inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T
    neural_network.train(training_set_inputs, training_set_outputs, 1000000)
    print("Pesos após o treino")
    neural_network.print_weights()
    print("Considerando uma nova situação [0, 0] -> 0: ")
    outputs = neural_network.think(array([0, 0]))
    print("Saida:\n")
    print(outputs[len(outputs)-1])
    print("Considerando uma nova situação [1, 0] -> 1: ")
    outputs = neural_network.think(array([1, 0]))
    print("Saida:\n")
    print(outputs[len(outputs)-1])

from numpy import exp, maximum, heaviside


class math_functions(object):
    def __init__(self):
        pass

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def dSigmoid(x):
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return maximum(x, 0)

    @staticmethod
    def dRelu(x):
        return 1. * (x > 0)  # + 0.1   * (x < 0) + 0

    @staticmethod
    def binary(x):
        if x > 0:
            y = 1
        else:
            y = 0
        return y

import numpy as np
import random

random.seed(0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1-sigmoid(x))


features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 x 2
features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 x 2
target_output = np.array([[0], [1], [1], [1]])
target_output = np.array([[0], [1], [1], [0]])
weights = np.array([[1.5], [-4.8]])  # 2x1
weights2 = np.array([[0.2]])  # 2x1
bias = 0.3
bias2 = 3
lr = 0.05


def train(inputs, weights, bias):
    perceptron_in = np.dot(inputs, weights) + bias
    perceptron_out = sigmoid(perceptron_in)
    error = perceptron_out - target_output

    sigmoid_der_res = sigmoid_der(perceptron_out)

    deriv = error * sigmoid_der_res

    weight_adjustments = np.dot(inputs.T, deriv)
    weights -= lr * weight_adjustments

    for i in deriv:
        bias -= lr * i

    return (weights, bias, perceptron_out)


print("Weights {}, Bias {}".format(weights, bias))

for epoch in range(10000):
    inputs = features
    weights, bias, layer1_out = train(inputs, weights, bias)
    weights2, bias2, layer2_out = train(layer1_out, weights2, bias2)


def predict(feature):
    layer1_out = sigmoid(np.dot(feature, weights) + bias)
    return sigmoid(np.dot(layer1_out, weights2) + bias2)


print("Weights {}, Bias {}".format(weights, bias))
print(predict(np.array([0, 0])))

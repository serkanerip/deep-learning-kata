import numpy as np

input_features = np.array([[0,0],[0,1],[1,0],[1,1]])
target_output = np.array([[0,1,1,1]])
target_output = target_output.reshape(4,1)
weights = np.array([[0.1],[0.2]])
lr = 0.05
bias = 0.3


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


def predict(feature):
    return sigmoid(np.dot(feature, weights) + bias)


for epoch in range(10000):
    inputs = input_features
    pred_in = np.dot(inputs, weights) + bias
    pred_out = sigmoid(pred_in)
    error = pred_out - target_output
    x = error.sum()

    dcost_dpred = error
    dpred_dz = sigmoid_der(pred_out)

    z_delta = dcost_dpred * dpred_dz
    inputs = input_features.T
    weights -= lr * np.dot(inputs, z_delta)

    for i in z_delta:
        bias -= lr * i

print(predict(np.array([1, 0])))

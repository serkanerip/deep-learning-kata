def sigmoid_act(x, der=False):
    import numpy as np

    if (der is True):
        f = 1/(1 + np.exp(- x))*(1-1/(1+ np.exp(- x)))
    else:
        f = 1/(1 + np.exp(- x))

    return f


def ReLU_act(x, der=False):
    import numpy as np

    if (der is True):
        f = np.heaviside(x, 1)
    else:
        f = np.maximum(x, 0)

    return f


def perceptron(X, act='Sigmoid'):
    import numpy as np
    shapes = X.shape
    n = shapes[0] + shapes[1]
    w = 2 * np.random.random(shapes) - 0.5
    b = np.random.random(1)

    f = b[0]

    for i in range(0, X.shape[0]-1):
        for j in range(0, X.shape[1]-1):
            f += w[i, j]*X[i, j]/n

    if act == 'Sigmoid':
        output = sigmoid_act(f)
    else:
        output = ReLU_act(f)

    return output


def ANN_train_orig(X_train, Y_train, p=4, q=4, eta=0.0015):
    import numpy as np
    import matplotlib.pyplot as plt

    w1 = 2*np.random.rand(p, X_train.shape[1]) - 0.5  # Layer 1
    b1 = np.random.rand(p)

    w2 = 2*np.random.rand(q, p) - 0.5  # Layer 2
    b2 = np.random.rand(q)

    wOut = 2*np.random.rand(q) - 0.5   # Output Layer
    bOut = np.random.rand(1)

    mu = []
    vec_y = []

    for index in range(0, X_train.shape[0] - 1):
        x = X_train[index]

        # Feed
        z1 = ReLU_act(np.dot(w1, x) + b1)  # output of layer1
        z2 = ReLU_act(np.dot(w2, z1) + b2)  # output of layer2
        y = sigmoid_act(np.dot(wOut, z2) + bOut)

        # Output layer error
        delta_out = (y - Y_train[index]) * sigmoid_act(y, der=True)

        delta_2 = delta_out * wOut * ReLU_act(z2, der=True)
        delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True)

        # Gradient Descent

        wOut -= eta * delta_out * z2
        bOut -= eta * delta_out

        w2 -= eta * np.kron(delta_2, z1).reshape(q, p)  # Hidden layer 2
        b2 -= eta * delta_2

        w1 -= eta * np.kron(delta_1, x).reshape(p, x.shape[0])  # Hidden layer1
        b1 -= eta * delta_1

        mu.append((y-Y_train[index])**2)
        vec_y.append(y)

    batch_loss = []
    for i in range(0, 10):
        loss_avg = 0
        for m in range(0, 60):
            loss_avg += vec_y[60*i+m]/60
        batch_loss.append(loss_avg)

    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(1, len(batch_loss)+1), batch_loss, alpha=1, s=10, label='error')
    plt.title('Averege Loss by epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()

    return w1, b1, w2, b2, wOut, bOut, mu

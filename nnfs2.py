import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funcs import perceptron, ReLU_act, sigmoid_act
from sklearn.model_selection import train_test_split


np.random.seed(10)

data = pd.read_csv('inputs/train.csv')


dict_live = {
    0: 'Perished',
    1: 'Survived'
}

dict_sex = {
    'male': 0,
    'female': 1   
}


data['Bsex'] = data['Sex'].apply(lambda x: dict_sex[x])
features = data[['Pclass', 'Bsex']].to_numpy()
labels = data['Survived'].to_numpy()

print(data.head(4))
print(features.shape)

print('Output with sigmoid activator: ', perceptron(features))
print('Output with ReLU activator: ', perceptron(features))

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.30)
print('Training records:', Y_train.size)
print('Test records:', Y_test.size)


def create_layers(layers_neurons=[4, 4]):
    layers = []
    idx = 0
    prev_layer_shape = X_train.shape[1]
    for i in layers_neurons:
        if idx - 1 > -1:
            prev_layer_shape = layers[idx - 1]['neurons']
        layers.append({
            'neurons': i,
            'weights': 2*np.random.rand(i, prev_layer_shape) - 0.5,
            'bias': np.random.rand(i),
            'delta': 0,
            'output': 0,
        })
        idx = idx + 1
    last_layer_neurons = layers[len(layers)-1]['neurons']

    layers.append({
        'neurons': last_layer_neurons,
        'weights': 2*np.random.rand(last_layer_neurons) - 0.5,
        'bias': np.random.rand(1),
        'delta': 0,
        'output': 0,
    })

    return layers


def ANN_train(X_train, Y_train, layers, eta=0.0015):
    import numpy as np
    import matplotlib.pyplot as plt
    mu = []
    vec_y = []

    for index in range(0, X_train.shape[0] - 1):
        x = X_train[index]

        # Feed
        layer_output = x
        for i in range(len(layers)):
            if i + 1 is len(layers):
                layer_output = sigmoid_act(np.dot(layers[i]['weights'], layer_output)) + layers[i]['bias']
            else:
                layer_output = ReLU_act(np.dot(layers[i]['weights'], layer_output) + layers[i]['bias'])
            layers[i]['output'] = layer_output
        y = layer_output

        # output layer error
        delta_Out = 2 * (y-Y_train[index]) * sigmoid_act(y, der=True)

        # calculate deltas
        for i in range(len(layers) - 2, -1, -1):
            if i is len(layers) - 2:
                layers[i]['delta'] = layers[len(layers)-1]['weights'] * delta_Out * ReLU_act(layers[i]['output'], der=True)
            else:
                layers[i]['delta'] = np.dot(layers[i+1]['delta'], layers[i+1]['weights']) * ReLU_act(layers[i]['output'], der=True)

        # adjust weights
        layers[len(layers) - 1]['weights'] -= eta * delta_Out * layers[len(layers) - 2]['output']
        layers[len(layers) - 1]['bias'] -= eta * delta_Out

        prev_layer_shape = x.shape[0]
        prev_layer_output = x
        for i in range(len(layers)-1):
            if i - 1 > -1:
                prev_layer_shape = layers[i-1]['neurons']
                prev_layer_output = layers[i-1]['output']
            weight = layers[i]['weights']
            delta = layers[i]['delta']
            neurons = layers[i]['neurons']
            layers[i]['weights'] -= eta * np.kron(delta, prev_layer_output).reshape(neurons, prev_layer_shape)
            layers[i]['bias'] -= eta * delta

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


layers = create_layers([8, 4, 1])
print(layers)
ANN_train(X_train, Y_train, layers)
print(layers)

# w1, b1, w2, b2, wOut, bOut, mu = ANN_train(X_train, Y_train, p=8, q=4)

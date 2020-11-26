* NN'de weightler değişiir. En iyi ağırlıklar bulunur.
* 
* İlk başta ağırlıklar random rakamlar olur daha sonra bunlar iyileştirilir.
* 



Adımlar:

1. Giriş verilerini al.
2. Bias'ı ekle.
3. Random ağırlıklar ekle feature'lara
4. Eğitim yap.
5. Hata oranını gör.
6. Hata oranına göre ağırlıkları değiştir.
7. Yeni ağırlıklar ile eğitimi tekrar yap.
8. Tahminleri yap.


## Perceptron

Perceptron, hidden layerı olmayan basit bir neural networktür. Sadece input ve output layerı vardır.

Percepton  lineer bir sınıflayıcıdır bu yüzden lineer bölünemeyen durumlarda sınıflama yapamaz. Örnek XOR.

![](https://miro.medium.com/max/318/0*Jti_QKi670rPEeq9.png)


![](https://miro.medium.com/max/423/0*vZ-zCjannuJrkZCl.png)

Bu durumdaki bir sorunu lineer olarak çözemeyiz parabolik bir fonksiyon ile çözebiliriz.

## Sigmoid Function

Sigmoid bir aktivasyon fonksiyonudur. NN'ler genellikle sınıflandırma için kullanılır. Binary sınıflandırmada 2 sınıf vardır. 
Ağırlıkların inputlarla çarpılıp toplanması ile çok büyük sayılar elde edebiliriz. Bu sorunu çözmek için bir aktivasyon fonksiyonu kullanırız.

Outputun 0 ve ya 1 olmasını isteriz bunun içinde sigmoid fonksiyonunu kullanırız. Sigmoid değerleri 0 ile 1 arasına çevirir.

![](https://miro.medium.com/max/624/0*85x58ShMCwJVfehZ.png)



# How to build a simple Neural Network with Python: Multi-layer Perceptron

A NN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. The basic example is the perceptron 

In ANN implementations, the "signal" at a connection is a real number, and the output of each neuron is computed by some non-linear function of the sum of its inputs.

A single layer perceptron (SLP) is a feed-forward network based on a threshold transfer function. SLP is the simplest type of artificial neural networks and can only classify linearly separable cases with a binary target (1, 0). [c], [d]

A multi-layer perceptron (MLP) has the same structure of a single layer perceptron with one or more hidden layers. The backpropagation algorithm consists of two phases: the forward phase where the activations are propagated from the input to the output layer, and the backward phase, where the error between the observed actual and the requested nominal value in the output layer is propagated backwards in order to modify the weights and bias values.


## Neural Network's Layer(s)

A standard Artificial Neural Network will be made of multiple layers:

1. An Input Layer, that pass the features to the NN
2. An arbitrary number of Hidden Layers, containing an arbitrary number of neurons for each layer, that receives the inputs and elaborate them. We will introduce Hidden Layers with ReLU activator, since in the hidden part of the NN we don't need the output to be contained in the [0,1] range.
3. An Output Layer: these layers contains a number of neurons equal to the number of possible labels we want to have a prediction to; this is because the output of the NN is thus a vector whose dimension is the same as the cardinality of the set of labels, and its entries are the probability for each label for the element whose feateures we have passed to the NN. This means that we will use a sigmoid activator to the Output layer, so we squeeze each perceptron's output between 0 and 1. 



## Links

1. https://pythonprogramming.net/machine-learning-tutorial-python-introduction/
2. https://www.kaggle.com/androbomb/simple-nn-with-python-multi-layer-perceptron
3. https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf
4. https://www.kaggle.com/uzairrj/beg-tut-intel-image-classification-93-76-accur/notebook
5. https://towardsdatascience.com/emulating-logical-gates-with-a-neural-network-75c229ec4cc9
6. https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf
7. https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/
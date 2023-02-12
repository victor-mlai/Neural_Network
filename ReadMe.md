# Neural Network
Learns using the input from the testing_data how to compute the output

## Using the program

In Main.cpp

* modify WHAT_TO_TRAIN to {COUNTER, IRIS, HAND_WRITTEN}
* play with the hidden layer sizes. Currently set to: "18, 30, 18"

## Results

### Bit Counter
```
Epoch: 10 - Loss: 0.720942 - Acc: 0.625

-------------------
Epoch: 20 - Loss: 0.682548 - Acc: 0.666667

-------------------
Epoch: 30 - Loss: 0.634227 - Acc: 0.75

-------------------
.....

-------------------
Epoch: 990 - Loss: 0.0175991 - Acc: 1

-------------------
Epoch: 1000 - Loss: 0.0166457 - Acc: 1

Training Time: 0.919937s

   network input   |    network output   |   target output
-------------------
0.000 0.000 0.000  | 0.045 0.034 0.988  | 0.000 0.000 1.000
-------------------
0.000 0.000 1.000  | 0.114 0.948 0.001  | 0.000 1.000 0.000
-------------------
0.000 1.000 0.000  | 0.119 0.979 0.997  | 0.000 1.000 1.000
-------------------
0.000 1.000 1.000  | 0.842 0.033 0.025  | 1.000 0.000 0.000
-------------------
1.000 0.000 0.000  | 0.981 0.043 0.977  | 1.000 0.000 1.000
-------------------
1.000 0.000 1.000  | 0.902 0.957 0.003  | 1.000 1.000 0.000
-------------------
1.000 1.000 0.000  | 0.896 0.938 0.993  | 1.000 1.000 1.000
-------------------
1.000 1.000 1.000  | 0.198 0.066 0.017  | 0.000 0.000 0.000
-------------------
```

### IRIS Dataset

```
-------------------
Epoch: 10 - Loss: 1.14499 - Acc: 0.555556

-------------------
Epoch: 20 - Loss: 0.952777 - Acc: 0.666667

-------------------
Epoch: 30 - Loss: 0.6809 - Acc: 0.773333

-------------------
.......

-------------------
Epoch: 970 - Loss: 0.050682 - Acc: 0.977778

-------------------
Epoch: 980 - Loss: 0.0677671 - Acc: 0.968889

-------------------
Epoch: 990 - Loss: 0.0909732 - Acc: 0.964444

-------------------
Epoch: 1000 - Loss: 0.0974372 - Acc: 0.957778

Training Time: 15.2231

   network input   |    network output   |   target output
-------------------
5.300 3.700 1.500 0.200  | 0.996 0.007 0.000  | 1.000 0.000 0.000
-------------------
5.000 3.300 1.400 0.200  | 0.996 0.007 0.000  | 1.000 0.000 0.000
-------------------
......
-------------------
5.100 2.500 3.000 1.100  | 0.015 0.999 0.000  | 0.000 1.000 0.000
-------------------
5.700 2.800 4.100 1.300  | 0.000 0.997 0.001  | 0.000 1.000 0.000
-------------------
......
-------------------
6.200 3.400 5.400 2.300  | 0.000 0.001 0.999  | 0.000 0.000 1.000
-------------------
5.900 3.000 5.100 1.800  | 0.000 0.001 0.998  | 0.000 0.000 1.000
-------------------
```

### Handwritten Digits Dataset

```
-------------------
Epoch: 1 - Loss: 0.693464 - Acc: 0.91906

-------------------
Epoch: 2 - Loss: 0.669076 - Acc: 0.92278

-------------------
Epoch: 3 - Loss: 0.624458 - Acc: 0.92644

-------------------
Epoch: 4 - Loss: 0.621624 - Acc: 0.9273

-------------------
Epoch: 5 - Loss: 0.606982 - Acc: 0.93016

-------------------
Epoch: 6 - Loss: 0.616215 - Acc: 0.92882

-------------------
Epoch: 7 - Loss: 0.615188 - Acc: 0.92914

-------------------
Epoch: 8 - Loss: 0.605424 - Acc: 0.93032

-------------------
Epoch: 9 - Loss: 0.596893 - Acc: 0.93118

-------------------
Epoch: 10 - Loss: 0.595913 - Acc: 0.93108

Training Time: 31.3381

# TODO: add network output
```

## Neural Network Algorithm
* is composed from multiple Layers: an Input Layer, 0 or more Hidden Layers and an Output Layer
* each Layer has a vector of Neurons

![](https://github.com/victorlaurentiu/Neural_Network/blob/master/NeuralNetwork.PNG)

* each Neuron has a vector of input weights and a bias value ( exception: the Input Layer )

![](https://github.com/victorlaurentiu/Neural_Network/blob/master/Neuron.PNG)

```
l - layer index
L - index of the output layer
n - neuron index
j - weight index
1 - one .. it has a pixel less than l

w[l][n] - Neuron's vector of weights
w[l][n][j] - the weight that connects the jth Neuron from the previous Layer with this one
b[l][n] - Neuron's bias value
```

### Feed Forward
```
a - each Neuron* has an activation output, and belongs to the interval (0, 1)

Set the corresponding output activations a[1] for the input layer.

For each layer l=2,3,…,L compute z[l] and a[l] for each Neuron

z = dot(a[l-1], w[l][n]) + b[l][n]		// dot product
a = s(z)

s(z) is a sigmoid function, sigmoid means it has an S form, for example:
s(z) = 1 / ( 1 + e^(-z) )
```

### Back Propagation
```
s'(z) - the derivative of s(z)

Compute the vector d[L], d - gradient
d[L] = (y - a[L]) * s'(z[L])

For each l=L-1,L-2,…,2 compute
d[l] = sumWGr(w[l+1], d[l+1]) * sigmoid'(z[l])
sumWGr(w[l+1], d[l+1]) = w[l+1][0][n] * d[l+1][0] + w[l+1][1][n] * d[l+1][1] * ...  
```

### Gradient Decent
```
For each Neuron update its weights and biases

deltaW = alpha * deltaW + eta * a[l-1] * d[l]
deltaB = alpha * deltaB + eta * d[l]

w += deltaW
b += deltaB
```

## To do:

* Optimize the algorithm even more ([Cross-Entropy](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function) and [Regularization](http://neuralnetworksanddeeplearning.com/chap3.html#regularization))

	[Improving the way neural networks learn](http://neuralnetworksanddeeplearning.com/chap3.html)
* Combine with the Genetic Algorithm to create a self driving car??? .. idk
* Recognize images with cucumbers


## References:
the book I used:

[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)

an youtube video of implementing this in C++ ... uses a little different approach than mine

[Neural Net in C++ Tutorial](https://www.youtube.com/watch?v=KkwX7FkLfug)

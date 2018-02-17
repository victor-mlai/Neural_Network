# Neural Network
Learns using the input from the testing_data how to compute the output

## Using the program
```
in the main()
	-write a series of inputs and their corresponding output (as an example check the sum of 2 bits)
	-create the Network ( using a topology vector{each number represents the number of Neurons in each Layer} ) 
	-and the number of iterations should be pretty high so set it right.
```

## Neural Network Algorithm
```
-is composed from multiple Layers: an Input Layer, 0 or more Hidden Layers and an Output Layer
-each Layer has a vector of Neuron*
-each Neuron has a vector of input weights and a bias value ( exception: the Input Layer )

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
```
	1. Optimize the algorithm even more (Cross-Entropy!!! and Regularization)
	http://neuralnetworksanddeeplearning.com/chap3.html	(Improving the way neural networks learn)
	2. Combine with the Genetic Algorithm to create a self driving car??? .. idk
	3. Recognize images with cucumbers
```

## References:
```
the book I used:
	http://neuralnetworksanddeeplearning.com/index.html (Neural Networks and Deep Learning)
	http://neuralnetworksanddeeplearning.com/chap3.html	(Improving the way neural networks learn)
an youtube video of implementing this in C++ ... uses a little different approach than mine
	https://www.youtube.com/watch?v=KkwX7FkLfug
```
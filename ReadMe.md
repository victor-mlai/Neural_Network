# Neural Network
Learns using the input from the testing_data how to compute the output

## Using the program
in the main() create 
```
	-the Network( using a topology vector{each number represents the number of Neurons in each Layer} ) 
	-write a series of inputs and their corresponding output
	-(as an example I let some commented code for a function)
	-and the number of iterations should be pretty high so set it right.
```

## Neural Network Algorithm
```
-It's composed from multiple Layers : an Input Layer, 0 or more Hidden Layers and an Output Layer
-each Layer has a vector of Neuron*
-each Neuron has a vector of input weights and a bias value ( exception: the Input Layer )

l - layer index
n - neuron index
j - weight index
1 - one .. it has a pixel less than l

w[l][n] - Neuron's vector of weights
w[l][n][j] - the weight that connects the jth Neuron from the previous Layer with this one
b[l][n] - Neuron's bias value

z = dot(a[l-1], w[l][n]) + b[l][n]		// dot product
a - each Neuron* has an activation output, and belongs to the interval (0, 1)
a = s(z)

s(z) is a sigmoid function, sigmoid means it has an S form, for example:
1 / ( 1 + e^(-z) )

Set the corresponding output activations a[1] for the input layer.
For each l=2,3,…,L compute z[l] and a[l]
Compute the vector d[L], d - gradient
s'(z) - the derivative of s(z)
d[L] = (y - a[L]) * s'(z[L])

For each l=L-1,L-2,…,2 compute
d[l] = sum-x (w[l+1], d[l+1]) * sigmoid'(z[l])

deltaW = alpha * deltaW + eta * a[l-1] * d[l]
deltaB = alpha * deltaB + eta * d[l]

w[l] += deltaW
b[l] += deltaB
```
### Feed Foward
It goes through every possible outcome and only chooses the best one, resulting in either a draw or a win for the AI player.

### Back Propagation

### Gradient Decent

## To do:
```
	1. Optimize the algorithm even more
	2. Recognize images with numbers
	3. Combine with the Genetic Algorithm to create a self driving car??? .. idk
```

## References:


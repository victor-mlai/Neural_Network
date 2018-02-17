#include "Network.h"
#include <time.h>

Network::Network()
{
}

/*
The list topo contains the number of neurons in the respective layers
	topo[0] = number of neurons in the input layer
	topo[1] = number of neurons in the 1st hidden layer
	...
	topo[len-1] = number of neurons in the output layer

The biases and weights in the Network object are all initialized randomly
*/
Network::Network(const vector<int>& topo) : topo(topo)
{
	srand(42);

	layers.resize(topo.size());
	for (int i = 1; i < topo.size(); i++) {
		layers[i].resize(topo[i]);
		for (int j = 0; j < topo[i]; j++) {
			layers[i][j] = new Neuron(topo[i - 1]);
		}
	}

	Layer& inputLayer = layers[0];
	inputLayer.resize(topo[0]);
	for (int j = 0; j < topo[0]; j++) {
		inputLayer[j] = new Neuron();
	}
}

Network::~Network()
{
	for (int i = 1; i < topo.size(); i++) {
		for (int j = 0; j < topo[i]; j++) {
			delete layers[i][j];
		}
	}
}

void Network::train(vector<float> in, vector<float> out)
{
	// vector<float> my_outs(getResults(in));
	// float Cost = RMS(my_outs, out);	// useless

	getResults(in);	// setting output activations

	// Calculate Gradients and update Weights and Biases
	{
		Layer& outLayer = layers.back();
		int j = 0;
		for (Neuron* n : outLayer) {
			n->calcOutGradient(out[j++]);
		}

		for (int i = layers.size() - 2; i > 0; i--) {
			Layer& currLayer = layers[i];
			Layer& nextLayer = layers[i + 1];
			int n_index = 0;	// neuron's index
			for (Neuron* n : currLayer) {
				n->calcHiddenGradient(nextLayer, n_index++);
			}
		}

		for (int i = layers.size() - 1; i > 0; i--) {
			Layer& currLayer = layers[i];
			Layer& prevLayer = layers[i - 1];
			for (Neuron* n : currLayer) {
				n->update(prevLayer);
			}
		}
	}
}

vector<float> Network::getResults(vector<float> in)
{
	vector<float> out(in);

	Layer& inputLayer = layers[0];
	for (int j = 0; j < topo[0]; j++) {
		inputLayer[j]->setActivation(in[j]);
	}
	for (int i = 1; i < topo.size(); i++) {	// for each Layer (except input layer)
		in = out;	// the input of this layer will be the output of the previous layer
		out.resize(topo[i]);
		int j = 0;
		for (Neuron* n : layers[i]) {	// for each Neuron in the current layer
			out[j++] = n->feedFoward(in);
		}
	}

	return out;
}

// Root Mean Square error
float Network::RMS(vector<float> target, vector<float> actual) {
	float rms = 0.0f;

	int n = target.size();
	for (int i = 0; i < n; i++) {
		float delta = target[i] - actual[i];
		rms += delta * delta;
	}

	return sqrt(rms / n);
}
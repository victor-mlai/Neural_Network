#include "Neuron.h"
#include <cmath> // in sigmoid function: used exp(x) <=> e^x
#include <cassert>	// in dot function: used assert()

const float Neuron::eta   = 0.2f;	// learning rate
const float Neuron::alpha = 0.6f;	// momentum constant

Neuron::Neuron()
{
}

Neuron::Neuron(int nr_inputs)
{
	// initializing bias and weights randomly between (-1.5 , 1.5)
	bias = rand() / (float)RAND_MAX * 3.0f - 1.5f;

	weights.resize(nr_inputs);
	for (int i = 0; i < nr_inputs; i++) {
		weights[i] = rand() / (float)RAND_MAX * 3.0f - 1.5f;
	}

	deltaBias = 0.0f;
	deltaWeights.resize(nr_inputs);
}

Neuron::~Neuron()
{
}

float Neuron::feedFoward(vector<float>& ins)
{
	z = dot(ins, weights) + bias;
	my_out = sigmoid(z);	// save value for back propagation
	return my_out;
}

void Neuron::calcOutGradient(float target)
{
	my_gradient = (target - my_out) * sigmoid_deriv(z);
}

void Neuron::calcHiddenGradient(const vector<Neuron*>& nextLayer, int my_index)
{
	my_gradient = sumWGr(nextLayer, my_index) * sigmoid_deriv(z);
}

void Neuron::update(vector<Neuron*>& prevLayer)
{
	for (int i = 0; i < weights.size(); i++) {
		deltaWeights[i] = ALPHA * deltaWeights[i] + ETA * prevLayer[i]->my_out * my_gradient;
		weights[i] += deltaWeights[i];
	}

	deltaBias = ALPHA * deltaBias + ETA * my_gradient;
	bias += deltaBias;
}

// ---------------------------------------------------
// Getters & Setters

float Neuron::getActivation() const
{
	return my_out;
}

void Neuron::setActivation(float out)
{
	my_out = out;
}

float Neuron::getGradient() const
{
	return my_gradient;
}

vector<float>& Neuron::getWeights()
{
	return weights;
}

// ---------------------------------------------------
// Helper functions

float Neuron::dot(const vector<float>& v1, const vector<float>& v2)
{
	assert(v1.size() == v2.size());

	float out = 0.0f;
	for (int i = 0; i < v1.size(); i++) {
		out += v1[i] * v2[i];
	}

	return out;
}

float Neuron::sigmoid(float x)
{
	return 1 / (1 + exp(-x)); // returns a value between (0, 1)
	// return tanh(x); // returns a value between (-1, 1)
}

float Neuron::sigmoid_deriv(float x)
{
	return sigmoid(x) * (1 - sigmoid(x));
	// return 1 - x * x; // a good aprox for tanh'(x)
}

// sum of w * gradient
float Neuron::sumWGr(const vector<Neuron*>& nextLayer, int my_index)
{
	float sum = 0.0f;

	for (Neuron* n : nextLayer) {
		// n->w[my_index] - the weight from "this" Neuron to "n" Neuron
		sum += n->getWeights()[my_index] * n->getGradient();
	}

	return sum;
}

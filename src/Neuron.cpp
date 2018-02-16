#include "Neuron.h"
#include <cmath> // in sigmoid function: used exp(x) <=> e^x
#include <cassert>	// in dot function: used assert()

float Neuron::eta = 0.02f;
float Neuron::alpha = 0.5f;

Neuron::Neuron()
{
}

Neuron::Neuron(int nr_inputs)
{
	// initializing bias and weights randomly
	bias = rand() % 20 - 10;

	weights.resize(nr_inputs);
	for (int i = 0; i < nr_inputs; i++) {
		weights[i] = rand() % 20 - 10;
	}

	deltaBias = 0.0f;
	deltaWeights.resize(nr_inputs);
}

Neuron::~Neuron()
{
}

float Neuron::feedFoward(vector<float>& ins)
{
	my_out = sigmoid(dot(ins, weights) + bias);	// save value for back propagation
	return my_out;
}

void Neuron::calcOutGradient(float target)
{
	my_gradient = (target - my_out) * sigmoid_deriv(my_out);
}

void Neuron::calcHiddenGradient(const vector<Neuron*>& nextLayer, int my_index)
{
	my_gradient = sumDow(nextLayer, my_index) * sigmoid_deriv(my_out);
}

void Neuron::update(vector<Neuron*>& prevLayer)
{
	for (int i = 0; i < weights.size(); i++) {
		deltaWeights[i] = deltaWeights[i] * alpha + eta * prevLayer[i]->my_out * my_gradient;
		weights[i] += deltaWeights[i];
	}

	deltaBias = deltaBias * alpha + eta * my_gradient;
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

float Neuron::sumDow(const vector<Neuron*>& nextLayer, int my_index)
{
	float sum = 0.0f;

	for (Neuron* n : nextLayer) {
		sum += n->getWeights()[my_index] * n->getGradient();
	}

	return sum;
}

#include "Network.hpp"
#include <random>
#include <cmath> // for exp
#include <cassert>	// for assert
#include <numeric>
#include <ranges>
#include <functional>

constexpr float ETA = 0.1f;  	// learning rate
constexpr float ALPHA = 0.9f;  	// momentum constant

namespace
{
float dot(const std::vector<float>& v1, const std::vector<float>& v2)
{
	assert(v1.size() == v2.size());

	return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.f);
}

// dot product between the weights of the connections that go in the Neuron
// from index "idx" in nextLayer with the Neurons' gradients
float DotProdBetw_Weigths_And_Gradients(const Layer& nextLayer, const size_t idx)
{
	float sum = 0.f;
	for (unsigned i = 0; i < nextLayer.m_nrOfNeurons; ++i)
	{
		sum += nextLayer.m_weights[i][idx] * nextLayer.m_gradients[i];
	}

	return sum;
}

// ----------------------------------------------------
// Sigmoid methods

float Sigmoid(float x)
{
	return 1 / (1 + exp(-x)); // returns a value between (0, 1)
	// return tanh(x); // returns a value between (-1, 1)
}

float SigmoidDeriv(float x)
{
	return Sigmoid(x) * (1 - Sigmoid(x));
	// return 1 - x * x; // a good aprox for tanh'(x)
}

float ReLu(float x)
{
	return x > 0.f ? x : 0.f;
}

float ReLuDeriv(float x)
{
	return x > 0.f ? 1.f : 0.f;
}
}

// ---------------------------------------------------
// Layer

Layer::Layer(const unsigned numNeurons, const unsigned numInputs) noexcept
	: m_nrOfNeurons(numNeurons),
	m_weights(numNeurons, std::vector<float>(numInputs)),
	m_dweights(numNeurons, std::vector<float>(numInputs)),
	m_biases(numNeurons),
	m_dbiases(numNeurons),
	m_activs(numNeurons),
	m_zs(numNeurons),
	m_gradients(numNeurons)
{
	std::mt19937 rng;
	std::uniform_real_distribution<float> distr(-1.f, 1.f);

	for (unsigned i = 0; i < m_nrOfNeurons; ++i)
	{
		std::generate_n(m_weights[i].begin(), numInputs, [&rng, &distr] { return distr(rng); });
	}
	std::generate_n(m_biases.begin(), numNeurons, [&rng, &distr] { return distr(rng); });
}

const std::vector<float>& Layer::FeedForward(const std::vector<float>& inputs)
{
	for (unsigned i = 0; i < m_nrOfNeurons; ++i)
	{
		m_zs[i] = dot(inputs, m_weights[i]) + m_biases[i];
		m_activs[i] = Sigmoid(m_zs[i]);	// save value for back propagation
	}

	return m_activs;
}

void Layer::CalcOutGradient(const std::vector<float>& y_train)
{
	for (unsigned i = 0; i < m_nrOfNeurons; ++i)
	{
		m_gradients[i] = (y_train[i] - m_activs[i]) * SigmoidDeriv(m_zs[i]);
	}
}

void Layer::CalcHiddenGradient(const Layer& nextLayer)
{
	for (unsigned i = 0; i < m_nrOfNeurons; ++i)
	{
		m_gradients[i] = DotProdBetw_Weigths_And_Gradients(nextLayer, i) * SigmoidDeriv(m_zs[i]);
	}
}

void Layer::update(const std::vector<float>& prevLayerActivations)
{
	for (unsigned i = 0; i < m_nrOfNeurons; ++i)
	{
		const size_t nrOfWeights = m_weights[i].size();
		for (size_t j = 0; j < nrOfWeights; ++j)
		{
			m_dweights[i][j] = ALPHA * m_dweights[i][j] + ETA * prevLayerActivations[j] * m_gradients[i];
			m_weights[i][j] += m_dweights[i][j];
		}

		m_dbiases[i] = ALPHA * m_dbiases[i] + ETA * m_gradients[i];
		m_biases[i] += m_dbiases[i];
	}
}

// ---------------------------------------------------
// Network

Network::Network(const std::vector<size_t>& layersSizes)
{
	srand(42);

	// make space for all layers except input layer
	m_layers.reserve(layersSizes.size() - 1);

	// add hidden layers + out layer
	for (unsigned i = 1; i < layersSizes.size(); ++i)
	{
		m_layers.emplace_back(
			/*nrOfNeurons: */ layersSizes[i],
			/*nrOfInput Connections: */ layersSizes[i - 1]);
	}
}

void Network::train(const std::vector<float>* x_train, const std::vector<float>& y_train)
{
	// Feed forward
	for (Layer& currLayer : m_layers)
	{
		x_train = &(currLayer.FeedForward(*x_train));
	}

	// Calculate Gradients
	{
		m_layers.back().CalcOutGradient(y_train);

		for (size_t j = m_layers.size() - 2; j > 0; j--)
		{
			Layer& currLayer = m_layers[j];
			const Layer& nextLayer = m_layers[j + 1];
			currLayer.CalcHiddenGradient(nextLayer);
		}
	}

	// Update Weights and Biases based on Calculated Gradients
	for (size_t i = m_layers.size() - 1; i > 1; i--)
	{
		Layer& currLayer = m_layers[i];
		const Layer& prevLayer = m_layers[i - 1];
		currLayer.update(prevLayer.m_activs);
	}
}

const std::vector<float>& Network::getResults(const std::vector<float>* in)
{
	for (Layer& currLayer : m_layers)
	{
		in = &(currLayer.FeedForward(*in));
	}

	return *in;
}

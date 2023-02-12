#pragma once

#include <vector>

class Layer
{
public:
	Layer(const unsigned numNeurons, const unsigned numInputs) noexcept;
	unsigned m_nrOfNeurons;
	std::vector< std::vector<float> > m_weights;
	std::vector< std::vector<float> > m_dweights;
	std::vector<float> m_biases;
	std::vector<float> m_dbiases;
	std::vector<float> m_activs;
	std::vector<float> m_zs;
	std::vector<float> m_gradients;
	const std::vector<float>& FeedForward(const std::vector<float>& inputs);
	void CalcOutGradient(const std::vector<float>& y_train);
	void CalcHiddenGradient(const Layer& nextLayer);
	void update(const std::vector<float>& prevLayerActivations);
};

// CostFunctions: Quadratic, Cross-Entropy
// Sigmoid, Tanh, ReLu, LeakyRelu
// Optimizers: None, Momentum, Nesterov, Adagrad, Adam

class Network
{
public:
	Network(const std::vector<size_t>& layersSizes);

	// Trains the network using the input vector and the output vector
	void train(const std::vector<float>* x_train, const std::vector<float>& y_train);
	
	// Trains the network using 
	//void StochasticGradientDecent(TrainingDataType& in, const int nrOfIter, const int batchSize);

	// Feed Forward the input vector
	const std::vector<float>& getResults(const std::vector<float>* in);

private:
	std::vector<Layer> m_layers;
};
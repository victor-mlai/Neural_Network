#pragma once

#include <vector>
using namespace std;

class Neuron {
public:
	Neuron();
	Neuron(int);
	~Neuron();
	float feedFoward(const vector<float>& ins);	// sets my_out using the input activations
	void calcOutGradient(float);	// Setting Output Layer Gradient
	void calcHiddenGradient(const vector<Neuron*>&, int);	// Setting Output Layer Gradient
	void update(const vector<Neuron*>&);

	float getActivation() const;
	void setActivation(float);
	float getGradient() const;
	vector<float>& getWeights();

private:
	float bias;
	float deltaBias;
	vector<float> weights;
	vector<float> deltaWeights;

	float my_out;		// activation output = sigmoid(z)
	float z;			// input for sigmoid(z)
	float my_gradient;	// error gradient

	//static const float eta;		// learning rate
	//static const float alpha;	// momentum constant

	#define ETA 0.2f	// learning rate
	#define ALPHA 0.6f // momentum constant

	// helper functions
	static float dot(const vector<float>& v1, const vector<float>& v2);	// dot product
	static float sigmoid(float x);
	static float sigmoid_deriv(float x);
	static float sumWGr(const vector<Neuron*>& nextLayer, int my_index);
};
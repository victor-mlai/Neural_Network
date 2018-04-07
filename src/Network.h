#pragma once
#include <vector>
#include "Neuron.h"
using namespace std;

typedef vector<Neuron*> Layer;

class Network {
public:
	Network();
	Network(const vector<int>&);
	~Network();

	// Trains the network using the input vector and the output vector
	void train(const vector<float> &in, const vector<float> &out);

	// Feed Forward the input vector
	vector<float> getResults(vector<float> in);

private:
	static float RMS(const vector<float> &target, const vector<float> &actual);

	vector<int> topo;	// topology
	vector<Layer> layers;
};
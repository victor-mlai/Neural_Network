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

	/*
	-Trains the network-
	feeds the in vector forward through the net
	computes the error, using the supposed output
	and then back propagates the error
	*/
	void train(vector<float> in, vector<float> out);

	/*
	*/
	vector<float> getResults(vector<float> in);

private:
	// Root Mean Square error
	float RMS(vector<float> target, vector<float> actual);

	vector<int> topo;	// topology
	vector<Layer> layers;
};


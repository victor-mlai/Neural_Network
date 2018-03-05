#include <iostream>
#include <vector>
#include "Network.h"

using namespace std;

void printSol(vector<float> net_in, vector<float> net_out, vector<float> out) {
	printf("in: ");
	for (float f : net_in) {
		printf("%.3f ", f);
	}
	printf(" | out: ");
	for (float f : out) {
		printf("%.3f ", f);
	}
	printf(" | net_out: ");
	for (float f : net_out) {
		printf("%.3f ", f);
	}
	printf("\n-------------------\n");
}

int main() {
	// Training the net
	// SUM - returns {sum, carry}
	vector< pair<vector<float>, vector<float>> > training_data =
	{
	//	{ {input}, {supposed output} }
		{ { 0, 0 }, {0, 0} },
		{ { 0, 1 }, {1, 0} },
		{ { 1, 0 }, {1, 0} },
		{ { 1, 1 }, {0, 1} },
	};

	int nr_inputs  = training_data[0].first.size();
	int nr_outputs = training_data[0].second.size();

	Network net({ nr_inputs, 3, nr_outputs });	// 1 hidden layer with 3 neurons

	int nrOfIter = 2000;	// number of iterations
	for (int i = 0; i < nrOfIter; i++)
	{
		for (auto t_data : training_data) {
			net.train(t_data.first, t_data.second);
		}
	}
	
	// Testing
	for (auto t_data : training_data) {
		printSol(t_data.first, net.getResults(t_data.first), t_data.second);
	}

	system("pause");
	return 0;
}
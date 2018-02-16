#include <iostream>
#include <vector>
#include "Network.h"

using namespace std;

void printSol(vector<float> net_in, vector<float> net_out, vector<float> out) {
	printf("in: ");
	for (float f : net_in) {
		printf("%f ", f);
	}
	printf("   |   net_out: ");
	for (float f : net_out) {
		printf("%f ", f);
	}
	printf("   |   out: ");
	for (float f : out) {
		printf("%f ", f);
	}
	printf("\n------------\n");
}

int main() {
	Network net({ 2, 3, 1 });

	// Training the net
	// XOR
	vector< pair<vector<float>, vector<float>> > training_data =
	{
	//	{ {input}, {supposed output} }
		{ { 0, 0 }, {0} },
		{ { 0, 1 }, {0} },
		{ { 1, 0 }, {1} },
		{ { 1, 1 }, {0} },
	};

	int nrOfIter = 10000;	// number of iterations
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
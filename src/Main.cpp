#include <iostream>
#include <vector>
#include "Network.h"
#include <chrono>	// for measuring the training time

using namespace std;

void printSol(const vector<float> &net_in, const vector<float> &net_out, const vector<float> &out) {
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
	// 3 bit counter
	vector< pair<vector<float>, vector<float>> > training_data =
	{
	//	{ {input}, {supposed output} }
		{ { 0, 0, 0 },{ 0, 0, 1 } },
		{ { 0, 0, 1 },{ 0, 1, 0 } },
		{ { 0, 1, 0 },{ 0, 1, 1 } },
		{ { 0, 1, 1 },{ 1, 0, 0 } },
		{ { 1, 0, 0 },{ 1, 0, 1 } },
		{ { 1, 0, 1 },{ 1, 1, 0 } },
		{ { 1, 1, 0 },{ 1, 1, 1 } },
		{ { 1, 1, 1 },{ 0, 0, 0 } }
	};

	int nr_inputs  = training_data[0].first.size();
	int nr_outputs = training_data[0].second.size();

	Network net({ nr_inputs, 8, 8, nr_outputs });	// 2 hidden layer with 8 neurons each

	int nrOfIter = 1000;	// number of iterations
	
	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < nrOfIter; i++) {
		for (auto t_data : training_data) {
			net.train(t_data.first, t_data.second);
		}
	}
	auto end = std::chrono::steady_clock::now();

	std::cout << "Training Time: " 
		<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0
		<< "\n";

	// Testing
	for (auto t_data : training_data) {
		printSol(t_data.first, net.getResults(t_data.first), t_data.second);
	}

	system("pause");
	return 0;
}

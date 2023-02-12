#include <iostream>
#include <vector>
#include <chrono>	// for measuring the training time
#include <fstream>
#include <string>
#include <iosfwd>
#include <sstream>
#include <cassert>
#include <ranges>

#include "Network.hpp"
#include "Utils.hpp"

#define COUNTER 0
#define IRIS 1
#define HAND_WRITTEN 2

#define WHAT_TO_TRAIN HAND_WRITTEN

#if WHAT_TO_TRAIN == COUNTER
constexpr int EPOCHS = 1000;
constexpr int TESTING_INTERVAL = 10;
#elif WHAT_TO_TRAIN == IRIS
constexpr int EPOCHS = 1000;
constexpr int TESTING_INTERVAL = 10;
#elif WHAT_TO_TRAIN == HAND_WRITTEN
constexpr int EPOCHS = 10;
constexpr int TESTING_INTERVAL = 1;
#endif

int main()
{
	const auto [x_train, y_train] =
#if WHAT_TO_TRAIN == COUNTER
		Utils::LoadBitCounterData();
#elif WHAT_TO_TRAIN == IRIS
		Utils::LoadIrisData();
#elif WHAT_TO_TRAIN == HAND_WRITTEN
		Utils::LoadHandWrittenData();
#endif
	assert(x_train.size() == y_train.size());

	const size_t nrInputs = x_train[0].size();
	const size_t nrOutputs = y_train[0].size();

	Network net({
		nrInputs,
#if WHAT_TO_TRAIN == COUNTER
		18, 30, 18,
#elif WHAT_TO_TRAIN == IRIS
		18, 30, 18,
#elif WHAT_TO_TRAIN == HAND_WRITTEN
		500, 50,
#endif
		nrOutputs });

	//-------------------------------------------
	auto start = std::chrono::steady_clock::now();

	for (unsigned i = 1; i < EPOCHS+1; i++)
	{
		for (unsigned j = 0; j < x_train.size(); j++)
			net.train(&x_train[j], y_train[j]);

		if (i % TESTING_INTERVAL == 0)
		{
			std::cout << "\n-------------------\n";
			const auto [mseLoss, acc] = Utils::CalcMSELossAndAccuracy(x_train, y_train, net);
			std::cout << "Epoch: " << i << " - Loss: " << mseLoss << " - Acc: " << acc << std::endl;
		}
	}

	auto end = std::chrono::steady_clock::now();
	//-------------------------------------------

	std::cout << "Training Time: " 
		<< std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0
		<< "\n";

	// Visualize
#if WHAT_TO_TRAIN != HAND_WRITTEN
	Utils::PrintSolutionForXOR(x_train, y_train, net);
#endif

	system("pause");
	return 0;
}

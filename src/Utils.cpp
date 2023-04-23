#include "Utils.hpp"
#include "Network.hpp"

#include <iostream>
#include <vector>
#include <chrono>	// for measuring the training time
#include <fstream>
#include <string>
#include <iosfwd>
#include <sstream>
#include <numeric>

namespace Utils
{
auto LoadBitCounterData() -> std::pair<X_Data_t, Y_Data_t>
{
	// 3 bit counter
	std::vector<std::vector<float>> x_train =
	{
		{ 0, 0, 0 },
		{ 0, 0, 1 },
		{ 0, 1, 0 },
		{ 0, 1, 1 },
		{ 1, 0, 0 },
		{ 1, 0, 1 },
		{ 1, 1, 0 },
		{ 1, 1, 1 }
	};

	std::vector<std::vector<float>> y_train =
	{
		{ 0, 0, 1 },
		{ 0, 1, 0 },
		{ 0, 1, 1 },
		{ 1, 0, 0 },
		{ 1, 0, 1 },
		{ 1, 1, 0 },
		{ 1, 1, 1 },
		{ 0, 0, 0 }
	};

	return std::pair(x_train, y_train);
}

auto LoadIrisData() -> std::pair<X_Data_t, Y_Data_t>
{
	const std::string fileName("./input_data/iris.data");
	std::ifstream inFile(fileName.c_str());
	if (!inFile)
	{
		printf("Could not open file: %s\n", fileName.c_str());
		exit(1);
	}

	X_Data_t x_train;
	Y_Data_t y_train;

	constexpr int nrOfInputs = 4;
	constexpr int nrOfOutputs = 3;

	for (std::string line; std::getline(inFile, line);)
	{
		std::istringstream inStr(line);

		std::vector<float> inputs;
		for (int j = 0; j < nrOfInputs; ++j)
		{
			float inVal; char comma;
			inStr >> inVal >> comma;
			inputs.push_back(inVal);
		}
		x_train.push_back(inputs);

		std::vector<float> outputs(nrOfOutputs);
		inStr >> line;
		if (line == "Iris-setosa")
		{
			outputs[0] = 1.0;
		}
		else if (line == "Iris-versicolor")
		{
			outputs[1] = 1.0;
		}
		else if (line == "Iris-virginica")
		{
			outputs[2] = 1.0;
		}
		else
		{
			printf("Unknown type %s.\n", line.c_str());
			exit(1);
		}
		y_train.push_back(outputs);

		printf("%f %f %f %f  ->   %f %f %f\n",
			inputs[0], inputs[1], inputs[2], inputs[3],
			outputs[0], outputs[1], outputs[2]);
	}

	return std::pair(x_train, y_train);
}

auto LoadHandWrittenData() -> std::pair<X_Data_t, Y_Data_t>
{
	// Load only a subset out of the 42.000 samples of 784 pixels.
	constexpr int max_num_samples = 5000;

	const std::string fileName("./input_data/handwritingData.txt");
	std::ifstream inFile(fileName.c_str());
	if (!inFile)
	{
		printf("Could not open file: %s\n", fileName.c_str());
		exit(1);
	}

	X_Data_t x_train;
	Y_Data_t y_train;
	constexpr int nrOfInputs = 784;
	constexpr int nrOfOutputs = 10;

	std::string line;
	while (getline(inFile, line))
	{
		std::istringstream s(std::move(line));
		int label; s >> label;
		std::vector<float> outputs(nrOfOutputs);
		outputs[label] = 1.f;

		std::vector<float> inputs(nrOfInputs);
		int colour;
		for (float& xIter : inputs)
		{
			s >> colour;
			xIter = (float)colour / 255.0f;
		}

		x_train.push_back(std::move(inputs));
		y_train.push_back(std::move(outputs));

		if (y_train.size() == max_num_samples)
			break;
	}

	return std::pair(x_train, y_train);
}

void PrintSolutionForXOR(
	const X_Data_t& x_train,
	const Y_Data_t& y_train,
	Network& net)
{
	printf("   network input   |    network output | target output   ");
	printf("\n-------------------\n");
	const size_t numSamples = x_train.size();
	for (size_t i = 0; i < numSamples; i++)
	{
		for (float in_f : x_train[i])
		{
			printf("%.3f ", in_f);
		}

		printf(" | ");

		const std::vector<float>& net_output = net.getResults(&x_train[i]);
		for (float n_out : net_output)
		{
			printf("%.3f ", n_out);
		}

		printf(" | ");

		for (float a_out : y_train[i])
		{
			printf("%.3f ", a_out);
		}

		printf("\n-------------------\n");
	}
}

auto CalcMSELossAndAccuracy(
	const X_Data_t& x_train,
	const Y_Data_t& y_train,
	Network& net) -> std::pair<float, float>
{
	float mseLoss = 0.f;
	size_t correctCount = 0;

	const size_t numSamples = x_train.size();
	for (size_t i = 0; i < numSamples; i++)
	{
		const std::vector<float>& net_output = net.getResults(&x_train[i]);

		const auto& target = y_train[i];

		const size_t outSize = target.size();
		for (size_t j = 0; j < outSize; j++)
		{
			const float delta = target[j] - net_output[j];
			mseLoss += delta * delta;
		}

		// count element-wise values in target and net_output are close
		correctCount += std::inner_product(target.begin(), target.end(), net_output.begin(), 0, std::plus<size_t>{},
			[](float t, float o) -> size_t { return (t > 0.5f && o > 0.5f) || (t < 0.5f && o < 0.5f); });
	}

	mseLoss /= numSamples;
	const float accuracy = (float)correctCount / (numSamples * y_train[0].size());

	return std::pair(mseLoss, accuracy);
}
}  // namespace Utils

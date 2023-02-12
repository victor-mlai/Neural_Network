#pragma once

#include <vector>

struct Network;

namespace Utils
{
using X_Data_t = std::vector<std::vector<float>>;
using Y_Data_t = std::vector<std::vector<float>>;

auto LoadBitCounterData() -> std::pair<X_Data_t, Y_Data_t>;

auto LoadIrisData() -> std::pair<X_Data_t, Y_Data_t>;

auto LoadHandWrittenData() -> std::pair<X_Data_t, Y_Data_t>;

void PrintSolutionForXOR(
	const X_Data_t& x_train,
	const Y_Data_t& y_train,
	Network& net);

auto CalcMSELossAndAccuracy(
	const X_Data_t& x_train,
	const Y_Data_t& y_train,
	Network& net) -> std::pair<float, float>;
}

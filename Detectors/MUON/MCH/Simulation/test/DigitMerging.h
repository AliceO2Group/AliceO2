#ifndef O2_MCH_SIMULATION_TEST_DIGITMERGING_H
#define O2_MCH_SIMULATION_TEST_DIGITMERGING_H

#include <vector>
#include "MCHSimulation/Digit.h"

std::vector<o2::mch::Digit> mergeDigitsMW(const std::vector<o2::mch::Digit>& inputDigits);

std::vector<o2::mch::Digit> mergeDigitsLA1(const std::vector<o2::mch::Digit>& inputDigits);

std::vector<o2::mch::Digit> mergeDigitsLA2(const std::vector<o2::mch::Digit>& inputDigits);

void dumpDigits(const std::vector<o2::mch::Digit>& digits);

using MergingFunctionType = std::vector<o2::mch::Digit> (*)(const std::vector<o2::mch::Digit>&);

std::vector<MergingFunctionType> mergingFunctions();

#endif

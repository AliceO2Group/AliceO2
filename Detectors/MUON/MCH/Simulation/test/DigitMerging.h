#ifndef O2_MCH_SIMULATION_TEST_DIGITMERGING_H
#define O2_MCH_SIMULATION_TEST_DIGITMERGING_H

#include <vector>
#include "MCHSimulation/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"

std::vector<o2::mch::Digit> mergeDigitsMW(const std::vector<o2::mch::Digit>& inputDigits, const std::vector<o2::MCCompLabel>& labels);

std::vector<o2::mch::Digit> mergeDigitsLA1(const std::vector<o2::mch::Digit>& inputDigits, const std::vector<o2::MCCompLabel>& labels);

std::vector<o2::mch::Digit> mergeDigitsLA2(const std::vector<o2::mch::Digit>& inputDigits, const std::vector<o2::MCCompLabel>& labels);

void dumpDigits(const std::vector<o2::mch::Digit>& digits);

using MergingFunctionType = std::vector<o2::mch::Digit> (*)(const std::vector<o2::mch::Digit>&, const std::vector<o2::MCCompLabel>&);

std::vector<MergingFunctionType> mergingFunctions();

#endif

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_SIMULATION_TEST_DIGITMERGING_H
#define O2_MCH_SIMULATION_TEST_DIGITMERGING_H

#include <vector>
#include "MCHBase/Digit.h"
#include "SimulationDataFormat/MCCompLabel.h"

std::vector<o2::mch::Digit> mergeDigitsMW(const std::vector<o2::mch::Digit>& inputDigits, const std::vector<o2::MCCompLabel>& labels);

std::vector<o2::mch::Digit> mergeDigitsLA1(const std::vector<o2::mch::Digit>& inputDigits, const std::vector<o2::MCCompLabel>& labels);

std::vector<o2::mch::Digit> mergeDigitsLA2(const std::vector<o2::mch::Digit>& inputDigits, const std::vector<o2::MCCompLabel>& labels);

void dumpDigits(const std::vector<o2::mch::Digit>& digits);

using MergingFunctionType = std::vector<o2::mch::Digit> (*)(const std::vector<o2::mch::Digit>&, const std::vector<o2::MCCompLabel>&);

std::vector<MergingFunctionType> mergingFunctions();

#endif

// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   Utils.h
/// @author SwirtaB
///

#ifndef O2_ZDC_FAST_SIMULATION_UTILS_H
#define O2_ZDC_FAST_SIMULATION_UTILS_H

#include <istream>
#include <random>
#include <vector>

namespace o2::zdc::fastsim
{
std::vector<float> normal_distribution(double mean, double stddev, size_t size);
std::vector<float> parse_block(std::istream& input, const std::string option);
} // namespace o2::zdc::fastsim
#endif // ZDC_FAST_SIMULATION_UTILS_H

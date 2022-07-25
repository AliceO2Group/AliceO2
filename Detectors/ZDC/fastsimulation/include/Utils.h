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
/**
 * @brief Generates a vector of numbers with a given normal distribution and length.
 *
 * @param mean normal distribution parameter
 * @param stddev normal distribution parameter
 * @param size size of generated vector
 * @return std::vector<float> vector of numbers
 */
std::vector<float> normal_distribution(double mean, double stddev, size_t size);

/**
 * @brief Generates a batch of vectors of numbers with a given normal distribution and length.
 *
 * @param mean normal distribution parameter
 * @param stddev normal distribution parameter
 * @param size size of generated vector
 * @param batchSize size of generated batch
 * @return std::vector<std::vector<float>>
 */
std::vector<std::vector<float>> normal_distribution_batch(double mean, double stddev, size_t size, long batchSize);

/**
 * @brief Parses .txt file containing scales for model.
 *        Function will search for given marker (option) and read 9 lines below that marker.
 *
 * @param input input stream (std::istream)
 * @param option marker (std::string, implementation recognize two markers: #means, #scales)
 * @return std::vector<float> loaded float numbers
 */
std::vector<float> parse_block(std::istream& input, const std::string& option);

/**
 * @brief
 *
 * @param vector2D
 * @return std::vector<float>
 */
std::vector<float> flat_vector(std::vector<std::vector<float>>& vector2D);

} // namespace o2::zdc::fastsim
#endif // O2_ZDC_FAST_SIMULATION_UTILS_H

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
/// @file   Utils.cxx
/// @author SwirtaB
///

#include "Utils.h"

std::vector<float> o2::zdc::fastsim::normal_distribution(double mean, double stddev, size_t size)
{
  // Creates 64bit random number generator
  std::random_device randomDevice{};
  std::mt19937_64 generator{randomDevice()};

  std::normal_distribution<> distribution{mean, stddev};

  // Each value in vector is generated with the same distribution
  std::vector<float> result;
  for (size_t i = 0; i < size; ++i) {
    result.emplace_back(distribution(generator));
  }
  return result;
}

std::vector<std::vector<float>> o2::zdc::fastsim::normal_distribution_batch(double mean,
                                                                            double stddev,
                                                                            size_t size,
                                                                            long batchSize)
{
  // Creates 64bit random number generator
  std::random_device randomDevice{};
  std::mt19937_64 generator{randomDevice()};

  std::normal_distribution<> distribution{mean, stddev};

  std::vector<std::vector<float>> batch;
  for (long i = 0; i < batchSize; ++i) {
    // Each value in vector is generated with the same distribution
    std::vector<float> result(size, 0);
    for (size_t i = 0; i < size; ++i) {
      result.at(i) = distribution(generator);
    }
    batch.emplace_back(std::move(result));
  }
  return batch;
}

std::vector<float> o2::zdc::fastsim::parse_block(std::istream& input, const std::string& option)
{
  // Marks if should start reading and parsing block
  bool read = false;
  // Container for parsed data (data) and current line (line)
  std::vector<float> data;
  std::string line;

  // read while eof if marker was found brake and start parsing
  while (std::getline(input, line)) {
    if (line == option) {
      read = true;
      break;
    }
  }

  // read while empty line or eof reached
  // discrads empty lines
  while (std::getline(input, line) && !line.empty()) {
    if (read) {
      data.push_back(std::stof(line));
    }
  }

  return data;
}

std::vector<float> o2::zdc::fastsim::flat_vector(std::vector<std::vector<float>>& vector2D)
{
  std::vector<float> flatVector;
  for (auto& vector1D : vector2D) {
    for (auto& value : vector1D) {
      flatVector.emplace_back(value);
    }
  }

  return flatVector;
}
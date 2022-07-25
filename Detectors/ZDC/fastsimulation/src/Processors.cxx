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
/// @file   Processors.cxx
/// @author SwirtaB
///

#include "Processors.h"

#include <cmath>

using namespace o2::zdc::fastsim::processors;

std::optional<std::vector<float>> StandardScaler::scale(const std::vector<float>& data) const
{
  if (data.size() != mMeans.size()) {
    return std::nullopt;
  }

  std::vector<float> scaledData(mMeans.size(), 0);
  for (size_t i = 0; i < mMeans.size(); ++i) {
    scaledData.at(i) = (data[i] - mMeans[i]) / mScales[i];
  }

  return scaledData;
}

std::optional<std::vector<std::vector<float>>> StandardScaler::scale_batch(
  const std::vector<std::vector<float>>& data) const
{

  std::vector<std::vector<float>> scaledData;
  for (auto& particleData : data) {
    if (particleData.size() != mMeans.size()) {
      return std::nullopt;
    }
    std::vector<float> scaledParticleData(mMeans.size(), 0);
    for (size_t i = 0; i < mMeans.size(); ++i) {
      scaledParticleData.at(i) = (particleData[i] - mMeans[i]) / mScales[i];
    }
    scaledData.emplace_back(std::move(scaledParticleData));
  }
  return scaledData;
}

bool StandardScaler::setScales(const std::vector<float>& means, const std::vector<float>& scales)
{
  if (means.size() != scales.size()) {
    return false;
  }
  mMeans = means;
  mScales = scales;
  return true;
}

std::vector<int> o2::zdc::fastsim::processors::readClassifier(const Ort::Value& value, const size_t batchSize)
{
  std::vector<int> results;
  auto predictedClasses = value.GetTensorData<float>();
  for (size_t i = 0; i < batchSize; ++i) {
    results.emplace_back(std::round(*(i + predictedClasses)));
  }
  return std::move(results);
}

std::vector<std::array<long, 5>> o2::zdc::fastsim::processors::calculateChannels(const Ort::Value& value,
                                                                                 const size_t batchSize)
{
  std::vector<std::array<long, 5>> results;               // results vector
  std::array<float, 5> channels = {0};                    // 5 photon channels
  auto flattedImageVector = value.GetTensorData<float>(); // Converts Ort::Value to flat const float*

  for (size_t batch = 0; batch < batchSize; ++batch) {
    // Model output needs to be converted with exp(x)-1 function to be valid
    for (int i = 0; i < 44; i++) {
      for (int j = 0; j < 44; j++) {
        if (i % 2 == j % 2) {
          if (i < 22 && j < 22) {
            channels[0] += std::expm1(flattedImageVector[j + i * 44 + (batch * 44 * 44)]);
          } else if (i < 22 && j >= 22) {
            channels[1] += std::expm1(flattedImageVector[j + i * 44 + (batch * 44 * 44)]);
          } else if (i >= 22 && j < 22) {
            channels[2] += std::expm1(flattedImageVector[j + i * 44 + (batch * 44 * 44)]);
          } else if (i >= 22 && j >= 22) {
            channels[3] += std::expm1(flattedImageVector[j + i * 44 + (batch * 44 * 44)]);
          }
        } else {
          channels[4] += std::expm1(flattedImageVector[j + i * 44 + (batch * 44 * 44)]);
        }
      }
    }
    results.emplace_back(std::array<long, 5>{0});
    for (int ch = 0; ch < 5; ++ch) {
      results.back()[ch] = std::lround(channels[ch]);
      channels[ch] = 0;
    }
  }
  return std::move(results);
}
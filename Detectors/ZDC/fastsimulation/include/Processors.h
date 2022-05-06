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
/// @file   Processors.h
/// @author SwirtaB
///

#ifndef O2_ZDC_FAST_SIMULATIONS_PROCESSORS_H
#define O2_ZDC_FAST_SIMULATIONS_PROCESSORS_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <optional>
#include <vector>

namespace o2::zdc::fastsim::processors
{

class StandardScaler
{
 public:
  StandardScaler() = default;
  ~StandardScaler() = default;

  /**
   * @brief Scales data with standard scale algorithm
   *
   * @param data
   * @return std::optional<std::vector<float>>
   */
  [[nodiscard]] std::optional<std::vector<float>> scale(const std::vector<float>& data) const;

  /**
   * @brief Scales batch of data with standard scale algorithm
   *
   * @param data
   * @return std::optional<std::vector<std::vector<float>>>
   */
  [[nodiscard]] std::optional<std::vector<std::vector<float>>> scale_batch(
    const std::vector<std::vector<float>>& data) const;

  /**
   * @brief Sets scales for standard scaler. Checks if sizes of scales are equal.
   *
   * @param means
   * @param scales
   * @return true on success
   * @return false on fail
   */
  bool setScales(const std::vector<float>& means, const std::vector<float>& scales);

 private:
  std::vector<float> mMeans;
  std::vector<float> mScales;
};

/**
 * @brief Reads predicted class as int
 *
 * @param value model prediction
 * @param batchSize
 * @return std::vector<int> casted results
 */
std::vector<int> readClassifier(const Ort::Value& value, size_t batchSize);

/**
 * @brief Calculate 5 channels values from 44x44 float array (for every batch)
 *
 * @param value model prediction
 * @param batchSize
 * @return std::vector<std::array<long, 5>> calculated results
 */
std::vector<std::array<long, 5>> calculateChannels(const Ort::Value& value, size_t batchSize);

} // namespace o2::zdc::fastsim::processors
#endif // O2_ZDC_FAST_SIMULATIONS_PROCESSORS_H
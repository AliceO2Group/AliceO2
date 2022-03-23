/**
 * @file Processors.h
 * @author SwirtaB
 * @brief
 * @version 0.1
 * @date 2022-03-19
 *
 * @copyright Copyright (c) 2022
 *
 */

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
   * @brief Scales data with standard scale aalgorithm
   *
   * @param data
   * @return std::optional<std::vector<float>>
   */
  std::optional<std::vector<float>> scale(const std::vector<float>& data) const;
  /**
   * @brief Sets scales for standard scaler. Checks if sizes of scales are equal.
   *
   * @param means
   * @param scales
   * @return true on success
   * @return false on fail
   */
  bool setScales(const std::vector<float> means, const std::vector<float> scales);

 private:
  std::vector<float> mMeans;
  std::vector<float> mScales;
};

/**
 * @brief Reads predicted class as int
 *
 * @param value model prediction
 * @return int predicted class
 */
int readClassifier(const Ort::Value& value);

/**
 * @brief Calculate 5 channels values from 44x44 float array
 *
 * @param value model prediction
 * @return std::array<long, 5> channels
 */
std::array<long, 5> calculateChannels(const Ort::Value& value);

} // namespace o2::zdc::fastsim::processors
#endif // O2_ZDC_FASTSIMULATIONS_PROCESSORS_H
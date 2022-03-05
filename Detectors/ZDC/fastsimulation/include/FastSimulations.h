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
/// @file   FastSimulations.h
/// @author SwirtaB
///

#ifndef O2_ZDC_FAST_SIMULATIONS_H
#define O2_ZDC_FAST_SIMULATIONS_H

#include "Config.h"
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <array>
#include <optional>

namespace o2::zdc::fastsim
{
std::array<int, 5> calculateChannels(Ort::Value& value);

class NeuralFastSimulation
{
 public:
  NeuralFastSimulation(const std::string& modelPath,
                       Ort::SessionOptions sessionOptions,
                       OrtAllocatorType allocatorType,
                       OrtMemType memoryType);
  ~NeuralFastSimulation() = default;

  virtual void run() = 0;
  virtual std::array<int, 5> getChannels() = 0;

 protected:
  void setInputOutputData();

  /// ONNX specific attributes
  Ort::Env mEnv;
  Ort::Session mSession;
  Ort::AllocatorWithDefaultOptions mAllocator;
  Ort::MemoryInfo mMemoryInfo;

  /// Input/Output names and input shape
  std::vector<char*> mInputNames;
  std::vector<char*> mOutputNames;
  std::vector<std::vector<int64_t>> mInputShapes;
};

class VAEModelSimulation : public NeuralFastSimulation
{
 public:
  VAEModelSimulation(std::array<float, 9>& conditionalMeans,
                     std::array<float, 9>& conditionalScales,
                     float noiseStdDev);
  ~VAEModelSimulation() = default;

  void run() override;
  std::array<int, 5> getChannels() override;
  void setData(std::array<float, 9>& particle);

 private:
  std::array<float, 9> scaleConditionalInput(const std::array<float, 9>& rawConditionalInput);

  std::array<float, 9> mConditionalMeans;
  std::array<float, 9> mConditionalScales;
  float mNoiseStdDev;
  std::vector<float> mNoiseInput;
  std::array<float, 9> mParticle{};
  std::vector<Ort::Value> mModelOutput;
};

class SAEModelSimulation : public NeuralFastSimulation
{
 public:
  SAEModelSimulation(std::array<float, 9>& conditionalMeans, std::array<float, 9>& conditionalScales);
  ~SAEModelSimulation() = default;

  void run() override;
  std::array<int, 5> getChannels() override;
  void setData(std::array<float, 9>& particle);

 private:
  std::array<float, 9> scaleConditionalInput(const std::array<float, 9>& rawConditionalInput);

  std::array<float, 9> mConditionalMeans;
  std::array<float, 9> mConditionalScales;
  std::vector<float> mNoiseInput;
  std::array<float, 9> mParticle{};
  std::vector<Ort::Value> mModelOutput;
};

std::optional<std::pair<std::array<float, 9>, std::array<float, 9>>> loadVaeScales(const std::string& path);
std::optional<std::pair<std::array<float, 9>, std::array<float, 9>>> loadSaeScales(const std::string& path);

} // namespace o2::zdc::fastsim
#endif // ONNX_API_FAST_SIMULATIONS_HPP
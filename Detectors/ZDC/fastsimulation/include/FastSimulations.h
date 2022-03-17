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

/**
 * @brief Abstract class providing interface for various specialized implementations.
 *
 */
class NeuralFastSimulation
{
 public:
  NeuralFastSimulation(const std::string& modelPath,
                       Ort::SessionOptions sessionOptions,
                       OrtAllocatorType allocatorType,
                       OrtMemType memoryType);
  ~NeuralFastSimulation() = default;

  /// Required interface
  /// setData(std::vector<float> &) - sets particle data, must check size of
  /// data run() - runs one simulation (single event), result should be stored as private member
  /// getChannels - caluclates 5 channels from result (stored as private member)
  /// second version should wrapp all function calls in one
  virtual bool setData(const std::vector<float>& particleData) = 0;
  virtual void run() = 0;
  virtual std::array<int, 5> getChannels() = 0;
  virtual std::array<int, 5> getChannels(const std::vector<float>& particleData) = 0;

 protected:
  /// Sets models metadata (input/output layers names, inputs shape) in onnx session
  void setInputOutputData();

  /// ONNX specific attributes
  /// User shoudn't has direct access to those in derived classes
  Ort::Env mEnv;
  Ort::Session mSession;
  Ort::AllocatorWithDefaultOptions mAllocator;
  Ort::MemoryInfo mMemoryInfo;

  /// Input/Output names and input shape
  std::vector<char*> mInputNames;
  std::vector<char*> mOutputNames;
  std::vector<std::vector<int64_t>> mInputShapes;
};

/**
 * @brief Derived class implementing interface for specific types of models.
 *
 */
class ConditionalModelSimulation : public NeuralFastSimulation
{
 public:
  ConditionalModelSimulation(const std::string& modelPath,
                             std::array<float, 9>& conditionalMeans,
                             std::array<float, 9>& conditionalScales,
                             float noiseStdDev);
  ~ConditionalModelSimulation() = default;

  void run() override;
  std::array<int, 5> getChannels() override;
  /**
   * @brief Set input data - particle information
   * Checks if data is right size - this check should be present in all implementations
   *
   * @param particle std::array<float, 9> with particle data
   * @return bool true if data is right size, false otherwise
   */
  bool setData(const std::vector<float>& particleData) override;
  /**
   * @brief Wraps three functions for convinience.
   *        It sets particle data, runs simulation, calculates channels and returns them.
   *
   * @param particle std::array<float, 9> with particle data
   * @return std::array<int, 5> calculated channels
   */
  std::array<int, 5> getChannels(const std::vector<float>& particleData) override;

 private:
  // Scales raw input using scales provided in seperate files
  // Works on the assumption that provided data are right size - this is checked in setData method
  std::array<float, 9> scaleConditionalInput(const std::vector<float>& rawConditionalInput);

  std::array<float, 9> mConditionalMeans;
  std::array<float, 9> mConditionalScales;
  float mNoiseStdDev;
  std::vector<float> mNoiseInput;
  std::vector<float> mParticle;
  std::vector<Ort::Value> mModelOutput;
};

std::optional<std::pair<std::array<float, 9>, std::array<float, 9>>> loadScales(const std::string& path);

class SimReaderBase
{
 public:
  virtual std::array<int, 5> visit(ConditionalModelSimulation& model) = 0;
};

class SimReader : public SimReaderBase
{
 public:
  std::array<int, 5> visit(ConditionalModelSimulation& model) override;
};

} // namespace o2::zdc::fastsim
#endif // ONNX_API_FAST_SIMULATIONS_HPP
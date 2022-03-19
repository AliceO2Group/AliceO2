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
/// @file   FastSimulations.cxx
/// @author SwirtaB
///

#include "FastSimulations.h"

#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <optional>

using namespace o2::zdc::fastsim;

std::array<int, 5> o2::zdc::fastsim::calculateChannels(Ort::Value& value)
{
  std::array<float, 5> channels = {0};                           // 5 photon channels
  auto flattedImageVector = value.GetTensorMutableData<float>(); // Converts Ort::Value to flat float*

  // Model output needs to be converted with exp(x)-1 function to be valid
  for (int i = 0; i < 44; i++) {
    for (int j = 0; j < 44; j++) {
      if (i % 2 == j % 2) {
        if (i < 22 && j < 22) {
          channels[0] += std::expm1(flattedImageVector[j + i * 44]);
        } else if (i < 22 && j >= 22) {
          channels[1] += std::expm1(flattedImageVector[j + i * 44]);
        } else if (i >= 22 && j < 22) {
          channels[2] += std::expm1(flattedImageVector[j + i * 44]);
        } else if (i >= 22 && j >= 22) {
          channels[3] += std::expm1(flattedImageVector[j + i * 44]);
        }
      } else {
        channels[4] += std::expm1(flattedImageVector[j + i * 44]);
      }
    }
  }
  std::array<int, 5> channels_integers = {0};
  for (int ch = 0; ch < 5; ch++) {
    channels_integers[ch] = std::round(channels[ch]);
  }
  return channels_integers;
}

//-------------------------------------------------NeuralFAstSimulation------------------------------------------------------

NeuralFastSimulation::NeuralFastSimulation(const std::string& modelPath,
                                           Ort::SessionOptions sessionOptions,
                                           OrtAllocatorType allocatorType,
                                           OrtMemType memoryType) : mSession(mEnv, modelPath.c_str(), sessionOptions),
                                                                    mMemoryInfo(Ort::MemoryInfo::CreateCpu(allocatorType, memoryType))
{
  setInputOutputData();
}

void NeuralFastSimulation::setInputOutputData()
{
  for (size_t i = 0; i < mSession.GetInputCount(); ++i) {
    mInputNames.push_back(mSession.GetInputName(i, mAllocator));
  }
  for (size_t i = 0; i < mSession.GetInputCount(); ++i) {
    mInputShapes.emplace_back(mSession.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
  }
  for (size_t i = 0; i < mSession.GetOutputCount(); ++i) {
    mOutputNames.push_back(mSession.GetOutputName(i, mAllocator));
  }

  // Prevent negative values from being passed as tensor shape
  // Which is no problem in python implementation of ONNX where -1 means that shape has to be figured out by library
  // In C++ this is illegal
  for (auto& shape : mInputShapes) {
    for (auto& elem : shape) {
      elem = std::abs(elem);
    }
  }
}

//---------------------------------------------------------Conditional-------------------------------------------------------
ConditionalModelSimulation::ConditionalModelSimulation(const std::string& modelPath,
                                                       std::array<float, 9>& conditionalMeans,
                                                       std::array<float, 9>& conditionalScales,
                                                       float noiseStdDev) : NeuralFastSimulation(modelPath, Ort::SessionOptions{nullptr}, OrtDeviceAllocator, OrtMemTypeCPU),
                                                                            mConditionalMeans(conditionalMeans),
                                                                            mConditionalScales(conditionalScales),
                                                                            mNoiseStdDev(noiseStdDev),
                                                                            mNoiseInput(normal_distribution(0, mNoiseStdDev, 10)) {}

void ConditionalModelSimulation::run()
{
  std::vector<Ort::Value> inputTensors;

  // Create tensor from noise
  inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
    mMemoryInfo, mNoiseInput.data(), mNoiseInput.size(), mInputShapes[0].data(), mInputShapes[0].size()));
  // Scale raw input and creates tensor from it
  auto conditionalInput = scaleConditionalInput(mParticle);
  inputTensors.emplace_back(Ort::Value::CreateTensor<float>(
    mMemoryInfo, conditionalInput.data(), conditionalInput.size(), mInputShapes[1].data(), mInputShapes[1].size()));

  // Run simulation (single event) with default run options
  mModelOutput = mSession.Run(Ort::RunOptions{nullptr},
                              mInputNames.data(),
                              inputTensors.data(),
                              inputTensors.size(),
                              mOutputNames.data(),
                              mOutputNames.size());
}

std::array<int, 5> ConditionalModelSimulation::getChannels()
{
  return calculateChannels(mModelOutput[0]);
}

void ConditionalModelSimulation::setData(const std::array<float, 9>& particleData)
{
  mParticle = particleData;
}

std::array<int, 5> ConditionalModelSimulation::getChannels(const std::array<float, 9>& particle)
{
  setData(particle);
  run();
  return calculateChannels(mModelOutput[0]);
}

std::array<float, 9> ConditionalModelSimulation::scaleConditionalInput(const std::array<float, 9>& rawConditionalInput)
{
  std::array<float, 9> scaledConditionalInput = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int i = 0; i < 9; ++i) {
    scaledConditionalInput[i] = (rawConditionalInput[i] - mConditionalMeans[i]) / mConditionalScales[i];
  }
  return scaledConditionalInput;
}

//---------------------------------------------------------Utils-------------------------------------------------------------
std::optional<std::pair<std::array<float, 9>, std::array<float, 9>>> o2::zdc::fastsim::loadScales(
  const std::string& path)
{
  std::fstream file(path, file.in);
  if (!file.is_open()) {
    return std::nullopt;
  }

  auto means = parse_block(file, "#means");
  if (means.size() != 9) {
    return std::nullopt;
  }

  auto scales = parse_block(file, "#scales");
  if (scales.size() != 9) {
    return std::nullopt;
  }

  std::array<float, 9> meansArray;
  std::array<float, 9> scalesArray;
  std::copy_n(std::make_move_iterator(means.begin()), 9, meansArray.begin());
  std::copy_n(std::make_move_iterator(scales.begin()), 9, scalesArray.begin());
  return std::make_pair(meansArray, scalesArray);
}

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

#include <fstream>

using namespace o2::zdc::fastsim;

//-------------------------------------------------NeuralFAstSimulation------------------------------------------------------

NeuralFastSimulation::NeuralFastSimulation(const std::string& modelPath,
                                           Ort::SessionOptions sessionOptions,
                                           OrtAllocatorType allocatorType,
                                           OrtMemType memoryType,
                                           int64_t batchSize) : mSession(mEnv, modelPath.c_str(), sessionOptions),
                                                                mMemoryInfo(Ort::MemoryInfo::CreateCpu(allocatorType, memoryType)),
                                                                mBatchSize(batchSize)
{
  setInputOutputData();
}

size_t NeuralFastSimulation::getBatchSize() const
{
  return mBatchSize;
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
  // Which is no problem in python implementation of ONNX where -1 means that
  // shape has to be figured out by library. In C++ this is illegal
  for (auto& shape : mInputShapes) {
    for (auto& elem : shape) {
      if (elem < 0) {
        elem = mBatchSize;
      }
    }
  }
}

void NeuralFastSimulation::setTensors(std::vector<std::vector<float>>& input)
{
  for (size_t i = 0; i < mInputShapes.size(); ++i) {
    mInputTensors.emplace_back(Ort::Value::CreateTensor<float>(
      mMemoryInfo, input[i].data(), input[i].size(), mInputShapes[i].data(), mInputShapes[i].size()));
  }
}

//---------------------------------------------------------Conditional-------------------------------------------------------
ConditionalModelSimulation::ConditionalModelSimulation(const std::string& modelPath, const int64_t batchSize) : NeuralFastSimulation(modelPath, Ort::SessionOptions{nullptr}, OrtDeviceAllocator, OrtMemTypeCPU, batchSize) {}

bool ConditionalModelSimulation::setInput(std::vector<std::vector<float>>& input)
{
  // Checks if number of inputs matches
  if (mSession.GetInputCount() != input.size()) {
    return false;
  }
  setTensors(input);
  return true;
}

void ConditionalModelSimulation::run()
{
  // Run simulation (single event) with default run options
  mModelOutput = mSession.Run(Ort::RunOptions{nullptr},
                              mInputNames.data(),
                              mInputTensors.data(),
                              mInputTensors.size(),
                              mOutputNames.data(),
                              mOutputNames.size());
  mInputTensors.clear();
}

const std::vector<Ort::Value>& ConditionalModelSimulation::getResult()
{
  return mModelOutput;
}

//------------------------------------------------------BatchHandler---------------------------------------------------------

BatchHandler::BatchHandler(size_t batchSize) : mBatchSize(batchSize) {}

BatchHandler& o2::zdc::fastsim::BatchHandler::getInstance(size_t batchSize)
{
  static o2::zdc::fastsim::BatchHandler instance(batchSize);
  return instance;
}

std::optional<std::vector<std::vector<float>>> BatchHandler::getBatch(const std::vector<float>& input)
{
  std::scoped_lock guard(mMutex);
  mBatch.emplace_back(input);
  if (mBatch.size() == mBatchSize) {
    auto value = std::optional(std::move(mBatch));
    mBatch.clear();
    return value;
  } else {
    return std::nullopt;
  }
}

//---------------------------------------------------------Utils-------------------------------------------------------------
std::optional<std::pair<std::vector<float>, std::vector<float>>> o2::zdc::fastsim::loadScales(const std::string& path)
{
  std::fstream file(path, std::fstream::in);
  if (!file.is_open()) {
    return std::nullopt;
  }

  auto means = parse_block(file, "#means");
  if (means.empty()) {
    return std::nullopt;
  }

  auto scales = parse_block(file, "#scales");
  if (scales.empty()) {
    return std::nullopt;
  }

  if (means.size() != scales.size()) {
    return std::nullopt;
  }
  return std::make_pair(std::move(means), std::move(scales));
}

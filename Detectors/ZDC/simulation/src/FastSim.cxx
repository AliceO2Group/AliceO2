// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ZDCSimulation/FastSim.h"
#include "Framework/Logger.h"
#include "DetectorsBase/SimRandom.h"
#include <TRandom.h>
#include <array>
#include <string>
#include <vector>
#include <cstring>

#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif

using namespace o2::zdc;

struct FastSim::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "ZDCFastSim"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<std::string> inputNames; // conditions, noise, (uniform)
  std::string outputName;
};

FastSim::FastSim() = default;
FastSim::~FastSim() = default;

namespace
{
// check that a tensor is [batch, n], with a dynamic or unit-compatible batch dimension
void checkShape(const Ort::TypeInfo& info, int n, const std::string& name)
{
  const auto tinfo = info.GetTensorTypeAndShapeInfo();
  const auto shape = tinfo.GetShape();
  if (tinfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || shape.size() != 2 || shape[1] != n) {
    LOG(fatal) << "ZDC fast simulation: tensor '" << name << "' must be a float tensor of shape [N," << n << "]";
  }
  if (shape[0] > 0) {
    LOG(fatal) << "ZDC fast simulation: tensor '" << name << "' has a fixed batch size " << shape[0] << ", a dynamic one is required";
  }
}
} // namespace

void FastSim::init(const FastSimModel& model)
{
  mModel = model;
  if (mModel.onnx.empty()) {
    LOG(fatal) << "ZDC fast simulation: the model does not contain any ONNX payload";
  }
  mImpl = std::make_unique<Impl>();
  // simulation workers are single threaded processes
  mImpl->options.SetIntraOpNumThreads(1);
  mImpl->options.SetInterOpNumThreads(1);
  mImpl->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  try {
    mImpl->session = std::make_unique<Ort::Session>(mImpl->env, mModel.onnx.data(), mModel.onnx.size(), mImpl->options);
  } catch (const Ort::Exception& e) {
    LOG(fatal) << "ZDC fast simulation: cannot create ONNX session: " << e.what();
  }
  auto& session = *mImpl->session;
  Ort::AllocatorWithDefaultOptions allocator;

  // inputs
  bool foundCond = false, foundNoise = false, foundUniform = false;
  for (size_t i = 0; i < session.GetInputCount(); i++) {
    const std::string name = session.GetInputNameAllocated(i, allocator).get();
    if (name == mModel.inputNameConditions) {
      checkShape(session.GetInputTypeInfo(i), FastSimModel::NConditions, name);
      foundCond = true;
    } else if (name == mModel.inputNameNoise) {
      checkShape(session.GetInputTypeInfo(i), FastSimModel::NOutputs, name);
      foundNoise = true;
    } else if (!mModel.inputNameUniform.empty() && name == mModel.inputNameUniform) {
      checkShape(session.GetInputTypeInfo(i), FastSimModel::NOutputs, name);
      foundUniform = true;
    } else {
      LOG(fatal) << "ZDC fast simulation: unexpected model input '" << name << "'";
    }
  }
  if (!foundCond || !foundNoise) {
    LOG(fatal) << "ZDC fast simulation: model inputs '" << mModel.inputNameConditions << "' and/or '" << mModel.inputNameNoise << "' not found";
  }
  mImpl->inputNames = {mModel.inputNameConditions, mModel.inputNameNoise};
  if (!mModel.inputNameUniform.empty()) {
    if (!foundUniform) {
      LOG(fatal) << "ZDC fast simulation: model input '" << mModel.inputNameUniform << "' not found";
    }
    mImpl->inputNames.push_back(mModel.inputNameUniform);
  }

  // output
  int outID = -1;
  for (size_t i = 0; i < session.GetOutputCount(); i++) {
    const std::string name = session.GetOutputNameAllocated(i, allocator).get();
    if (mModel.outputName.empty() || name == mModel.outputName) {
      outID = i;
      mImpl->outputName = name;
      break;
    }
  }
  if (outID < 0) {
    LOG(fatal) << "ZDC fast simulation: model output '" << mModel.outputName << "' not found";
  }
  checkShape(session.GetOutputTypeInfo(outID), FastSimModel::NOutputs, mImpl->outputName);
  LOG(info) << "ZDC fast simulation initialized";
  mModel.print();
}

void FastSim::run(const std::vector<float>& conditions, int nParticles, std::vector<float>& photons)
{
  photons.clear();
  if (nParticles <= 0) {
    return;
  }
  if (!mImpl) {
    LOG(fatal) << "ZDC fast simulation used without initialization";
  }
  if (conditions.size() != size_t(nParticles) * FastSimModel::NConditions) {
    LOG(fatal) << "ZDC fast simulation: inconsistent size of input " << conditions.size() << " for " << nParticles << " particles";
  }
  auto* rnd = o2::base::simRandom(); // the generator of the transport of this thread
  mNoise.resize(size_t(nParticles) * FastSimModel::NOutputs);
  for (auto& v : mNoise) {
    v = rnd->Gaus(0., 1.);
  }
  const std::array<int64_t, 2> shapeCond{nParticles, FastSimModel::NConditions}, shapeNoise{nParticles, FastSimModel::NOutputs};
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(Ort::Value::CreateTensor<float>(mImpl->memInfo, const_cast<float*>(conditions.data()), conditions.size(), shapeCond.data(), shapeCond.size()));
  inputs.emplace_back(Ort::Value::CreateTensor<float>(mImpl->memInfo, mNoise.data(), mNoise.size(), shapeNoise.data(), shapeNoise.size()));
  if (mImpl->inputNames.size() > 2) {
    // uniform numbers for the stochastic rounding done inside the model
    mUniform.resize(mNoise.size());
    for (auto& v : mUniform) {
      v = rnd->Rndm();
    }
    inputs.emplace_back(Ort::Value::CreateTensor<float>(mImpl->memInfo, mUniform.data(), mUniform.size(), shapeNoise.data(), shapeNoise.size()));
  }
  std::vector<const char*> inNames;
  for (const auto& n : mImpl->inputNames) {
    inNames.push_back(n.c_str());
  }
  const std::array<const char*, 1> outNames{mImpl->outputName.c_str()};
  try {
    auto out = mImpl->session->Run(Ort::RunOptions{nullptr}, inNames.data(), inputs.data(), inputs.size(), outNames.data(), outNames.size());
    const auto nOut = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (nOut != size_t(nParticles) * FastSimModel::NOutputs) {
      LOG(fatal) << "ZDC fast simulation: unexpected output size " << nOut << " for " << nParticles << " particles";
    }
    photons.resize(nOut);
    std::memcpy(photons.data(), out[0].GetTensorData<float>(), nOut * sizeof(float));
  } catch (const Ort::Exception& e) {
    LOG(fatal) << "ZDC fast simulation: inference failed: " << e.what();
  }
}

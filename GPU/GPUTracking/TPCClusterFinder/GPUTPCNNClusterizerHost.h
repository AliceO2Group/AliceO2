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

/// \file GPUTPCNNClusterizerHost.h
/// \author Christian Sonnabend

#ifndef O2_GPUTPCNNCLUSTERIZERHOST_H
#define O2_GPUTPCNNCLUSTERIZERHOST_H

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
#ifdef GPUCA_HAS_ONNX
#include "ML/OrtInterface.h"
#endif

class OrtMemoryInfo;
class OrtAllocator;
struct MockedOrtAllocator;
namespace Ort
{
struct Env;
struct MemoryInfo;
} // namespace Ort

namespace o2::OrtDataType
{
struct Float16_t;
}

namespace o2::gpu
{

class GPUReconstruction;
class GPUTPCNNClusterizer;
struct GPUSettingsProcessingNNclusterizer;

class GPUTPCNNClusterizerHost
{
 public:
  GPUTPCNNClusterizerHost() = default;
  GPUTPCNNClusterizerHost(const GPUSettingsProcessingNNclusterizer& settings, bool useDeterministicMode = false) { init(settings, useDeterministicMode); }

  void init(const GPUSettingsProcessingNNclusterizer&, bool = false);
  void initClusterizer(const GPUSettingsProcessingNNclusterizer&, GPUTPCNNClusterizer&, int32_t = -1, int32_t = -1);
  void createBoundary(GPUTPCNNClusterizer&);
  void createIndexLookup(GPUTPCNNClusterizer&);

  void initSofie(const GPUSettingsProcessingNNclusterizer&, void* stream, int32_t device, bool hip, const GPUTPCNNClusterizerHost* source = nullptr);
  void useSofie(const GPUTPCNNClusterizerHost&);
  void bindSofieWorkspace(void* workspace, size_t bytes);
  void inferenceSofie(int model, const void* input, size_t batch, void* output);
  int32_t modelOutputs(int model) const;
  template <class Input, class Output>
  void inference(int model, Input* input, size_t batch, Output* output)
  {
    if (mSofie) {
      inferenceSofie(model, input, batch, output);
      return;
    }
#ifdef GPUCA_HAS_ONNX
    (model == 0 ? mModelClass : model == 1 ? mModelReg1
                                           : mModelReg2)
      .inference(input, batch, output);
#else
    throw std::runtime_error("ONNXRuntime was not enabled in this build");
#endif
  }

#ifdef GPUCA_HAS_ONNX
  // ONNX
  void directOrtAllocator(Ort::Env*, Ort::MemoryInfo*, GPUReconstruction*, bool = false);
  MockedOrtAllocator* getMockedAllocator();
  const OrtMemoryInfo* getMockedMemoryInfo();

  std::unordered_map<std::string, std::string> mOrtOptions;
  o2::ml::OrtModel mModelClass, mModelReg1, mModelReg2;  // For splitting clusters
  std::shared_ptr<MockedOrtAllocator> mMockedAlloc = nullptr;
#endif
  std::vector<bool> mModelsUsed = {false, false, false}; // 0: class, 1: reg_1, 2: reg_2
  int32_t mDeviceId = -1;

 private:
  struct SofieState;
  std::shared_ptr<SofieState> mSofie;
}; // class GPUTPCNNClusterizerHost

} // namespace o2::gpu

#endif

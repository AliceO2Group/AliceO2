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

/// \file GPUTPCNNClusterizerHost.cxx
/// \author Christian Sonnabend

#ifdef GPUCA_HAS_SOFIE
#include "Rtypes.h"
#endif

#include <CommonUtils/StringUtils.h>

#include "GPUTPCNNClusterizerHost.h"
#include "GPUTPCNNClusterizer.h"
#include "GPUSettings.h"
#include "ML/3rdparty/GPUORTFloat16.h"
#include "GPUReconstruction.h"
#include "GPULogging.h"
#include "GPUTPCGeometry.h"
#include "DataFormatsTPC/Constants.h"
#include "clusterFinderDefs.h"

#ifdef GPUCA_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#ifdef GPUCA_HAS_SOFIE
#include <TMVA/RGPUModel.hxx>
#include <TMVA/RModelParser_ONNX.hxx>
#endif
#include <array>
#include <limits>
#include <sstream>

using namespace o2::gpu;

struct GPUTPCNNClusterizerHost::SofieState {
#ifdef GPUCA_HAS_SOFIE
  std::array<std::shared_ptr<TMVA::Experimental::SOFIE::RGPUModel>, 3> models;
  std::array<std::unique_ptr<TMVA::Experimental::SOFIE::RGPUModel::Session>, 3> sessions;
#endif
  std::array<std::string, 3> buffers;
  size_t workspaceSize = 0;
  unsigned int maxBatch = 0;
};

void GPUTPCNNClusterizerHost::init(const GPUSettingsProcessingNNclusterizer& settings, bool useDeterministicMode)
{
#ifndef GPUCA_HAS_ONNX
  throw std::runtime_error("ONNXRuntime was not enabled in this build");
#else
  std::string class_model_path = settings.nnClassificationPath, reg_model_path = settings.nnRegressionPath;
  std::vector<std::string> reg_model_paths_local;
  std::vector<std::string> evalMode = o2::utils::Str::tokenize(settings.nnEvalMode, ':');

  if (settings.nnLoadFromCCDB) {
    reg_model_path = settings.nnLocalFolder + "/net_regression_c1.onnx"; // Needs to be set identical to GPUWorkflowSpec.cxx, otherwise the networks might be loaded from the wrong place
    if (evalMode[0] == "c1") {
      class_model_path = settings.nnLocalFolder + "/net_classification_c1.onnx";
    } else if (evalMode[0] == "c2") {
      class_model_path = settings.nnLocalFolder + "/net_classification_c2.onnx";
    }

    if (evalMode[1] == "r2") {
      reg_model_path += ":" + settings.nnLocalFolder + "/net_regression_c2.onnx";
    }
  }

  mOrtOptions = {
    {"model-path", class_model_path},
    {"device-type", settings.nnInferenceDevice},
    {"allocate-device-memory", std::to_string(settings.nnInferenceAllocateDevMem)},
    {"intra-op-num-threads", std::to_string(settings.nnInferenceIntraOpNumThreads)},
    {"inter-op-num-threads", std::to_string(settings.nnInferenceInterOpNumThreads)},
    {"enable-optimizations", std::to_string(settings.nnInferenceEnableOrtOptimization)},
    {"deterministic-compute", std::to_string(useDeterministicMode ? 1 : settings.nnInferenceUseDeterministicCompute)}, // TODO: This unfortunately doesn't guarantee determinism (25.07.2025)
    {"enable-profiling", std::to_string(settings.nnInferenceOrtProfiling)},
    {"profiling-output-path", settings.nnInferenceOrtProfilingPath},
    {"logging-level", std::to_string(settings.nnInferenceVerbosity)},
    {"onnx-environment-name", "c1"}};

  mModelClass.initOptions(mOrtOptions);
  mModelsUsed[0] = true;

  reg_model_paths_local = o2::utils::Str::tokenize(reg_model_path, ':');

  if (!settings.nnClusterizerUseCfRegression) {
    if (reg_model_paths_local.size() == 1) {
      mOrtOptions["model-path"] = reg_model_paths_local[0];
      mOrtOptions["onnx-environment-name"] = "r1";
      mModelReg1.initOptions(mOrtOptions);
      mModelsUsed[1] = true;
    } else {
      mOrtOptions["model-path"] = reg_model_paths_local[0];
      mOrtOptions["onnx-environment-name"] = "r1";
      mModelReg1.initOptions(mOrtOptions);
      mModelsUsed[1] = true;
      mOrtOptions["model-path"] = reg_model_paths_local[1];
      mOrtOptions["onnx-environment-name"] = "r2";
      mModelReg2.initOptions(mOrtOptions);
      mModelsUsed[2] = true;
    }
  }
#endif
}

void GPUTPCNNClusterizerHost::initClusterizer(const GPUSettingsProcessingNNclusterizer& settings, GPUTPCNNClusterizer& clustererNN, int32_t maxFragmentLen, int32_t maxAllowedTimebin)
{
  clustererNN.mNnClusterizerUseCfRegression = settings.nnClusterizerUseCfRegression;
  clustererNN.mNnClusterizerSizeInputRow = settings.nnClusterizerSizeInputRow;
  clustererNN.mNnClusterizerSizeInputPad = settings.nnClusterizerSizeInputPad;
  clustererNN.mNnClusterizerSizeInputTime = settings.nnClusterizerSizeInputTime;
  clustererNN.mNnClusterizerFullRowSize = 2 * settings.nnClusterizerSizeInputRow + 1;
  clustererNN.mNnClusterizerFullPadSize = 2 * settings.nnClusterizerSizeInputPad + 1;
  clustererNN.mNnClusterizerFullTimeSize = 2 * settings.nnClusterizerSizeInputTime + 1;
  clustererNN.mNnClusterizerChargeArraySize = clustererNN.mNnClusterizerFullRowSize * clustererNN.mNnClusterizerFullPadSize * clustererNN.mNnClusterizerFullTimeSize;
  clustererNN.mNnClusterizerPadTimeSize = clustererNN.mNnClusterizerFullPadSize * clustererNN.mNnClusterizerFullTimeSize;
  clustererNN.mNnClusterizerRowTimeSize = clustererNN.mNnClusterizerFullRowSize * clustererNN.mNnClusterizerFullTimeSize;
  clustererNN.mNnClusterizerRowTimeSizeFull = clustererNN.mNnClusterizerRowTimeSize + (settings.nnClusterizerAddIndexData ? 3 : 0);
  clustererNN.mNnClusterizerRowTimeSizeThreads = clustererNN.mNnClusterizerRowTimeSize + (settings.nnClusterizerAddIndexData ? 1 : 0);
  clustererNN.mNnClusterizerElementSize = clustererNN.mNnClusterizerChargeArraySize + (settings.nnClusterizerAddIndexData ? 3 : 0);
  // clustererNN.mBoundaryMapSizeRow = 3 * clustererNN.mNnClusterizerSizeInputRow + o2::tpc::constants::MAXGLOBALPADROW;
  // clustererNN.mBoundaryPadding = 11; // padding on each side to account for pad_offset. N=11 since then mIsBoundary = 24320 ~< (1.5 x 2^14 = 24576) && N must be bigger than (NPads[row(end_iroc + 1)] - NPads[row(end_iroc)])/2 (=6) for pad_offset to work
  // clustererNN.mBoundaryMapSizePadsPerRow = GPUTPCGeometry::NPads(o2::tpc::constants::MAXGLOBALPADROW - 1) + 2 * clustererNN.mBoundaryPadding;
  // clustererNN.mBoundaryMapSize = clustererNN.mBoundaryMapSizeRow * clustererNN.mBoundaryMapSizePadsPerRow;
  // clustererNN.mIndexLookupSize = 3 * clustererNN.mNnClusterizerChargeArraySize; // local row, pad, time shift from flat index
  clustererNN.mNnClusterizerAddIndexData = settings.nnClusterizerAddIndexData;
  clustererNN.mNnClusterizerBatchedMode = settings.nnClusterizerBatchedMode;
  clustererNN.mNnClusterizerBoundaryFillValue = settings.nnClusterizerBoundaryFillValue;
  clustererNN.mNnSigmoidTrafoClassThreshold = settings.nnSigmoidTrafoClassThreshold;
  clustererNN.mNnClusterizerUseClassification = settings.nnClusterizerUseClassification;
  clustererNN.mNnClusterizerSetDeconvolutionFlags = (bool)settings.nnClusterizerSetDeconvolutionFlags;
  clustererNN.maxFragmentLen = maxFragmentLen == -1 ? TPC_MAX_FRAGMENT_LEN_GPU : maxFragmentLen;
  clustererNN.maxAllowedTimebin = maxAllowedTimebin == -1 ? TPC_MAX_FRAGMENT_LEN_GPU : maxAllowedTimebin;
  if (clustererNN.mNnSigmoidTrafoClassThreshold) {
    clustererNN.mNnClassThreshold = (float)std::log(settings.nnClassThreshold / (1.f - settings.nnClassThreshold));
  } else {
    clustererNN.mNnClassThreshold = settings.nnClassThreshold;
  }
  if (settings.nnClusterizerVerbosity < 0) {
    clustererNN.mNnClusterizerVerbosity = settings.nnInferenceVerbosity;
  } else {
    clustererNN.mNnClusterizerVerbosity = settings.nnClusterizerVerbosity;
  }
  // Define the datatype for input and output
  if (settings.nnInferenceInputDType.find("32") != std::string::npos) {
    clustererNN.mNnInferenceInputDType = 0;
  } else {
    clustererNN.mNnInferenceInputDType = 1; // Default to float16
  }
  if (settings.nnInferenceOutputDType.find("32") != std::string::npos) {
    clustererNN.mNnInferenceOutputDType = 0;
  } else {
    clustererNN.mNnInferenceOutputDType = 1; // Default to float16
  }
#ifdef GPUCA_HAS_SOFIE
  if (mSofie) {
    if (settings.nnClusterizerBatchedMode != mSofie->maxBatch) {
      throw std::runtime_error("Changing SOFIE batch capacity requires chain reinitialization");
    }
    clustererNN.mSofieWorkspaceSize = mSofie->workspaceSize;
    for (const auto& model : mSofie->models) {
      if (model && ((model->GetPrecision() == TMVA::Experimental::SOFIE::RGPUModel::Precision::Float16) != (clustererNN.mNnInferenceInputDType == 1) || clustererNN.mNnInferenceInputDType != clustererNN.mNnInferenceOutputDType)) {
        throw std::runtime_error("Changing SOFIE precision requires chain reinitialization");
      }
      if (model && model->InputSize() != static_cast<size_t>(clustererNN.mNnClusterizerElementSize)) {
        throw std::runtime_error("SOFIE input shape does not match the configured clusterizer window");
      }
    }
  }
#endif
  clustererNN.mNnClusterizerModelClassNumOutputNodes = modelOutputs(0);
  if (!settings.nnClusterizerUseCfRegression) {
    if (modelOutputs(0) == 1 || !mModelsUsed[2]) {
      clustererNN.mNnClusterizerModelReg1NumOutputNodes = modelOutputs(1);
    } else {
      clustererNN.mNnClusterizerModelReg1NumOutputNodes = modelOutputs(1);
      clustererNN.mNnClusterizerModelReg2NumOutputNodes = modelOutputs(2);
    }
  }
}

// void GPUTPCNNClusterizerHost::createBoundary(GPUTPCNNClusterizer& clustererNN)
// {
//   // Call after init of the clustererNN elements
//   for (int r = 0; r < clustererNN.mBoundaryMapSizeRow; r++) {
//     int8_t skipCheckInRow = 0;
//     for (int p = 0; p < clustererNN.mBoundaryMapSizePadsPerRow; p++) {
//       int32_t i = r * clustererNN.mBoundaryMapSizePadsPerRow + p;
//       clustererNN.mIsBoundary[i] = 1;
//       if (!skipCheckInRow && (p >= clustererNN.mBoundaryPadding || r >= clustererNN.mNnClusterizerSizeInputRow)) {
//         if (r < (GPUTPCGeometry::EndIROC() + clustererNN.mNnClusterizerSizeInputRow)) {
//           clustererNN.mIsBoundary[i] = (int32_t)((p - clustererNN.mBoundaryPadding) >= static_cast<int>(GPUTPCGeometry::NPads(r - clustererNN.mNnClusterizerSizeInputRow)));
//         } else if (r >= (GPUTPCGeometry::EndIROC() + 2 * clustererNN.mNnClusterizerSizeInputRow) && r < (o2::tpc::constants::MAXGLOBALPADROW + 2 * clustererNN.mNnClusterizerSizeInputRow)) {
//           clustererNN.mIsBoundary[i] = (int32_t)((p - clustererNN.mBoundaryPadding) >= static_cast<int>(GPUTPCGeometry::NPads(r - 2 * clustererNN.mNnClusterizerSizeInputRow)));
//         }
//         skipCheckInRow = (clustererNN.mIsBoundary[i] == 1); // No need to check further pads in this row
//       }
//     }
//   }
// }

// void GPUTPCNNClusterizerHost::createIndexLookup(GPUTPCNNClusterizer& clustererNN)
// {
//   for (int32_t i = 0; i < clustererNN.mNnClusterizerChargeArraySize; i++) {
//     int32_t r = CAMath::Floor(i / ((2 * clustererNN.mNnClusterizerSizeInputPad + 1) * (2 * clustererNN.mNnClusterizerSizeInputTime + 1))) - clustererNN.mNnClusterizerSizeInputRow;
//     int32_t rest_1 = i % ((2 * clustererNN.mNnClusterizerSizeInputPad + 1) * (2 * clustererNN.mNnClusterizerSizeInputTime + 1));
//     int32_t p = CAMath::Floor(rest_1 / (2 * clustererNN.mNnClusterizerSizeInputTime + 1)) - clustererNN.mNnClusterizerSizeInputPad;
//     int32_t t = (rest_1 % (2 * clustererNN.mNnClusterizerSizeInputTime + 1)) - clustererNN.mNnClusterizerSizeInputTime;
//     clustererNN.mIndexLookup[3 * i] = r;
//     clustererNN.mIndexLookup[3 * i + 1] = p;
//     clustererNN.mIndexLookup[3 * i + 2] = t;
//   }
// }

#ifdef GPUCA_HAS_ONNX
// MockedOrtAllocator implementation to be able to use volatile assignment
struct MockedOrtAllocator : OrtAllocator {
  MockedOrtAllocator(GPUReconstruction* = nullptr, OrtMemoryInfo* = nullptr);
  ~MockedOrtAllocator();

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo* Info() const;
  void* Reserve(size_t size);
  size_t NumAllocations() const;
  size_t NumReserveAllocations() const;

  void LeakCheck();

 private:
  MockedOrtAllocator(const MockedOrtAllocator&) = delete;
  MockedOrtAllocator& operator=(const MockedOrtAllocator&) = delete;

  std::atomic<size_t> memory_inuse{0};
  std::atomic<size_t> num_allocations{0};
  std::atomic<size_t> num_reserve_allocations{0};
  OrtMemoryInfo* mMemoryInfoInternal;
  GPUReconstruction* mRecInternal;
};

MockedOrtAllocator::MockedOrtAllocator(GPUReconstruction* r, OrtMemoryInfo* info)
{
  OrtAllocator::version = ORT_API_VERSION;
  OrtAllocator::Alloc = [](OrtAllocator* this_, size_t size) { return static_cast<MockedOrtAllocator*>(this_)->Alloc(size); };
  OrtAllocator::Free = [](OrtAllocator* this_, void* p) { static_cast<MockedOrtAllocator*>(this_)->Free(p); };
  OrtAllocator::Info = [](const OrtAllocator* this_) { return static_cast<const MockedOrtAllocator*>(this_)->Info(); };
  OrtAllocator::Reserve = [](OrtAllocator* this_, size_t size) { return static_cast<MockedOrtAllocator*>(this_)->Reserve(size); };
  mRecInternal = r;
  mMemoryInfoInternal = info;
}

MockedOrtAllocator::~MockedOrtAllocator()
{
  // Ort::GetApi().ReleaseMemoryInfo(mMemoryInfoInternal);
  (void)0; // Suppress warning for empty destructor
}

void* MockedOrtAllocator::Alloc(size_t size)
{
  LOG(info) << "(ORT) Allocating direct memory of size " << size << " bytes";
  return mRecInternal->AllocateDirectMemory(size, GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_STACK);
}

void* MockedOrtAllocator::Reserve(size_t size)
{
  LOG(info) << "(ORT) Reserving direct memory of size " << size << " bytes";
  return mRecInternal->AllocateDirectMemory(size, GPUMemoryResource::MEMORY_GPU | GPUMemoryResource::MEMORY_STACK);
}

void MockedOrtAllocator::Free(void* p)
{
  // LOG(info) << "(ORT) Freeing volatile memory " << p;
}

const OrtMemoryInfo* MockedOrtAllocator::Info() const
{
  return mMemoryInfoInternal;
}

size_t MockedOrtAllocator::NumAllocations() const
{
  return num_allocations.load();
}

size_t MockedOrtAllocator::NumReserveAllocations() const
{
  return num_reserve_allocations.load();
}

void MockedOrtAllocator::LeakCheck()
{
  if (memory_inuse.load()) {
    LOG(warning) << "memory leak!!!";
  }
}

void GPUTPCNNClusterizerHost::directOrtAllocator(Ort::Env* env, Ort::MemoryInfo* memInfo, GPUReconstruction* rec, bool recreate)
{
  mMockedAlloc = std::make_shared<MockedOrtAllocator>(rec, (OrtMemoryInfo*)(*memInfo));
  if (recreate) {
    Ort::ThrowOnError(Ort::GetApi().UnregisterAllocator((OrtEnv*)(*env), (OrtMemoryInfo*)(*memInfo)));
  }
  Ort::ThrowOnError(Ort::GetApi().RegisterAllocator((OrtEnv*)(*env), mMockedAlloc.get()));
  memInfo = (Ort::MemoryInfo*)mMockedAlloc->Info();
}

const OrtMemoryInfo* GPUTPCNNClusterizerHost::getMockedMemoryInfo()
{
  return mMockedAlloc->Info();
}

MockedOrtAllocator* GPUTPCNNClusterizerHost::getMockedAllocator()
{
  return mMockedAlloc.get();
}

#endif

void GPUTPCNNClusterizerHost::initSofie(const GPUSettingsProcessingNNclusterizer& settings, void* stream, int32_t device, bool hip, const GPUTPCNNClusterizerHost* source, const std::array<std::string_view, 3>& buffers, const GPUTPCNNClusterizerHost* previous)
{
#ifdef GPUCA_HAS_SOFIE
  using Model = TMVA::Experimental::SOFIE::RGPUModel;
  if (!settings.nnClusterizerBatchedMode || settings.nnClusterizerBatchedMode > static_cast<unsigned>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("SOFIE requires a positive batch capacity within int32 range");
  }
  auto precision = [](const std::string& value) {
    if (value == "FP32" || value == "fp32") {
      return Model::Precision::Float32;
    }
    if (value == "FP16" || value == "fp16") {
      return Model::Precision::Float16;
    }
    throw std::runtime_error("SOFIE precision must be FP32 or FP16");
  };
  const auto inputType = precision(settings.nnInferenceInputDType);
  if (inputType != precision(settings.nnInferenceOutputDType)) {
    throw std::runtime_error("SOFIE requires matching model, input and output precision");
  }
  if (settings.nnInferenceDevice != (hip ? "rocm" : "cuda") && settings.nnInferenceDevice != (hip ? "ROCM" : "CUDA")) {
    throw std::runtime_error("SOFIE inference device must match the reconstruction backend (cuda or rocm)");
  }
  size_t inputSize = 1;
  for (int radius : {settings.nnClusterizerSizeInputRow, settings.nnClusterizerSizeInputPad, settings.nnClusterizerSizeInputTime}) {
    if (radius < 0 || radius > 16383 || inputSize > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / (2 * radius + 1)) {
      throw std::runtime_error("SOFIE clusterizer input window is too large or invalid");
    }
    inputSize *= 2 * radius + 1;
  }
  inputSize += settings.nnClusterizerAddIndexData ? 3 : 0;
  if (inputSize > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / settings.nnClusterizerBatchedMode) {
    throw std::runtime_error("SOFIE input batch exceeds the clusterizer's index range");
  }
  auto state = std::make_shared<SofieState>();
  state->maxBatch = settings.nnClusterizerBatchedMode;
  std::array<std::string, 3> paths{settings.nnClassificationPath, "", ""};
  if (settings.nnLoadFromCCDB) {
    auto mode = o2::utils::Str::tokenize(settings.nnEvalMode, ':');
    if (mode.size() != 2 || (mode[0] != "c1" && mode[0] != "c2") || (mode[1] != "r1" && mode[1] != "r2")) {
      throw std::runtime_error("SOFIE CCDB loading requires nnEvalMode c1:r1, c1:r2, c2:r1 or c2:r2");
    }
    paths[0] = "CCDB classification";
    if (!settings.nnClusterizerUseCfRegression) {
      paths[1] = "CCDB regression 1";
      if (mode[1] == "r2") {
        paths[2] = "CCDB regression 2";
      }
    }
  } else if (!settings.nnClusterizerUseCfRegression) {
    auto regression = o2::utils::Str::tokenize(settings.nnRegressionPath, ':');
    if (regression.empty() || regression.size() > 2) {
      throw std::runtime_error("SOFIE expects one or two regression paths");
    }
    for (size_t i = 0; i < regression.size(); i++) {
      paths[i + 1] = regression[i];
    }
  }
  TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
  std::unordered_map<std::string, std::shared_ptr<Model>> compiled;
  if (settings.nnLoadFromCCDB && previous && previous->mSofie) {
    for (size_t i = 0; i < previous->mSofie->models.size(); i++) {
      if (previous->mSofie->models[i] && !previous->mSofie->buffers[i].empty()) {
        compiled.emplace(previous->mSofie->buffers[i], previous->mSofie->models[i]);
      }
    }
  }
  for (size_t i = 0; i < paths.size(); i++) {
    mModelsUsed[i] = !paths[i].empty();
    if (!mModelsUsed[i]) {
      continue;
    }
    if (source) {
      state->models[i] = source->mSofie->models[i];
    } else {
      if (settings.nnLoadFromCCDB && buffers[i].empty()) {
        throw std::runtime_error("Missing or empty ONNX buffer for " + paths[i]);
      }
      auto& model = compiled[settings.nnLoadFromCCDB ? std::string(buffers[i]) : paths[i]];
      if (settings.nnLoadFromCCDB) {
        state->buffers[i] = buffers[i];
      }
      if (!model) {
        GPUInfo("SOFIE: preparing model %zu (%s), backend=%s, architecture=%s, precision=%s",
                i, paths[i].c_str(), hip ? "HIP" : "CUDA", settings.sofieArchitecture.c_str(),
                inputType == Model::Precision::Float16 ? "FP16" : "FP32");
        if (settings.nnLoadFromCCDB) {
          std::istringstream input(state->buffers[i], std::ios::in | std::ios::binary);
          model = std::make_shared<Model>(parser.ParseGPU(input));
        } else {
          model = std::make_shared<Model>(parser.ParseGPU(paths[i]));
        }
        if (model->GetPrecision() != inputType) {
          throw std::runtime_error("SOFIE model precision differs from nnInferenceInputDType: " + paths[i]);
        }
        if (model->InputSize() != inputSize || model->OutputSize() > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / settings.nnClusterizerBatchedMode) {
          throw std::runtime_error("SOFIE model shape does not match the clusterizer window or index range: " + paths[i]);
        }
        model->WorkspaceSize(settings.nnClusterizerBatchedMode);
        model->Compile({hip ? Model::Backend::HIP : Model::Backend::CUDA, settings.sofieCompiler, settings.sofieArchitecture});
        GPUInfo("SOFIE: model %zu compilation succeeded, input=%zu, output=%zu",
                i, model->InputSize(), model->OutputSize());
      } else {
        GPUInfo("SOFIE: model %zu reuses an existing compiled program", i);
      }
      state->models[i] = model;
    }
    state->sessions[i] = state->models[i]->CreateSession(stream, device, settings.nnClusterizerBatchedMode);
    if (state->sessions[i]->WorkspaceSize() > std::numeric_limits<size_t>::max() - state->workspaceSize) {
      throw std::overflow_error("SOFIE combined workspace size overflow");
    }
    state->workspaceSize += state->sessions[i]->WorkspaceSize();
  }
  mSofie = std::move(state);
  mDeviceId = device;
#else
  throw std::runtime_error("SOFIE was not enabled in this build");
#endif
}

bool GPUTPCNNClusterizerHost::hasSofieBuffers(const std::array<std::string_view, 3>& buffers) const
{
  if (!mSofie) {
    return false;
  }
  for (size_t i = 0; i < buffers.size(); i++) {
    if (mModelsUsed[i] && buffers[i] != mSofie->buffers[i]) {
      return false;
    }
  }
  return true;
}

void GPUTPCNNClusterizerHost::useSofie(const GPUTPCNNClusterizerHost& source)
{
  mSofie = source.mSofie;
  mModelsUsed = source.mModelsUsed;
  mDeviceId = source.mDeviceId;
}

void GPUTPCNNClusterizerHost::bindSofieWorkspace(void* workspace, size_t bytes)
{
#ifdef GPUCA_HAS_SOFIE
  if (!mSofie) {
    return;
  }
  if (!workspace || bytes < mSofie->workspaceSize) {
    throw std::runtime_error("O2 supplied insufficient SOFIE workspace");
  }
  auto* next = static_cast<char*>(workspace);
  for (const auto& session : mSofie->sessions) {
    if (session) {
      session->SetWorkspace(next, session->WorkspaceSize());
      next += session->WorkspaceSize();
    }
  }
#endif
}

void GPUTPCNNClusterizerHost::inferenceSofie(int model, const void* input, size_t batch, void* output)
{
#ifdef GPUCA_HAS_SOFIE
  if (!mSofie || model < 0 || model >= 3 || !mSofie->sessions[model]) {
    throw std::runtime_error("SOFIE model is not initialized");
  }
  mSofie->sessions[model]->Infer(input, output, batch);
#else
  throw std::runtime_error("SOFIE was not enabled in this build");
#endif
}

int32_t GPUTPCNNClusterizerHost::modelOutputs(int model) const
{
#ifdef GPUCA_HAS_SOFIE
  if (mSofie) {
    return mSofie->models.at(model) ? static_cast<int32_t>(mSofie->models.at(model)->OutputSize()) : 0;
  }
#endif
#ifdef GPUCA_HAS_ONNX
  return (model == 0 ? mModelClass : model == 1 ? mModelReg1
                                                : mModelReg2)
    .getNumOutputNodes()
    .at(0)
    .at(1);
#else
  throw std::runtime_error("No initialized inference backend");
#endif
}

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

#include <CommonUtils/StringUtils.h>

#include "GPUTPCNNClusterizerHost.h"
#include "GPUTPCNNClusterizer.h"
#include "GPUSettings.h"
#include "ML/3rdparty/GPUORTFloat16.h"
#include "GPUReconstruction.h"
#include "GPUTPCGeometry.h"
#include "DataFormatsTPC/Constants.h"

#ifdef GPUCA_HAS_ONNX
#include <onnxruntime_cxx_api.h>
#endif

using namespace o2::gpu;

void GPUTPCNNClusterizerHost::init(const GPUSettingsProcessingNNclusterizer& settings, bool useDeterministicMode)
{
  std::string class_model_path = settings.nnClassificationPath, reg_model_path = settings.nnRegressionPath;
  std::vector<std::string> reg_model_paths_local;
  std::vector<std::string> evalMode = o2::utils::Str::tokenize(settings.nnEvalMode, ':');

  if (settings.nnLoadFromCCDB) {
    reg_model_path = settings.nnLocalFolder + "/net_regression_c1.onnx"; // Needs to be set identical to NeuralNetworkClusterizer.cxx, otherwise the networks might be loaded from the wrong place
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
}

void GPUTPCNNClusterizerHost::initClusterizer(const GPUSettingsProcessingNNclusterizer& settings, GPUTPCNNClusterizer& clustererNN)
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
  clustererNN.mNnInferenceInputDType = settings.nnInferenceInputDType.find("32") != std::string::npos;
  clustererNN.mNnInferenceOutputDType = settings.nnInferenceOutputDType.find("32") != std::string::npos;
  clustererNN.mNnClusterizerModelClassNumOutputNodes = mModelClass.getNumOutputNodes()[0][1];
  if (!settings.nnClusterizerUseCfRegression) {
    if (mModelClass.getNumOutputNodes()[0][1] == 1 || !mModelReg2.isInitialized()) {
      clustererNN.mNnClusterizerModelReg1NumOutputNodes = mModelReg1.getNumOutputNodes()[0][1];
    } else {
      clustererNN.mNnClusterizerModelReg1NumOutputNodes = mModelReg1.getNumOutputNodes()[0][1];
      clustererNN.mNnClusterizerModelReg2NumOutputNodes = mModelReg2.getNumOutputNodes()[0][1];
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

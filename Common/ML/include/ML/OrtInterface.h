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

/// \file     OrtInterface.h
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
/// \brief    A header library for loading ONNX models and inferencing them on CPU and GPU

#ifndef O2_ML_ORTINTERFACE_H
#define O2_ML_ORTINTERFACE_H

// C++ and system includes
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <unordered_map>

// O2 includes
#include "GPUCommonLogger.h"

namespace Ort
{
struct SessionOptions;
struct MemoryInfo;
struct Env;
} // namespace Ort

namespace o2::ml
{

class OrtModel
{

 public:
  // Constructors & destructors
  OrtModel();
  OrtModel(std::unordered_map<std::string, std::string> optionsMap);
  void init(std::unordered_map<std::string, std::string> optionsMap);
  virtual ~OrtModel();

  // General purpose
  void initOptions(std::unordered_map<std::string, std::string> optionsMap);
  void initEnvironment();
  void initSession();
  void memoryOnDevice(int32_t = 0);
  bool isInitialized() { return mInitialized; }
  void resetSession();

  // Getters
  std::vector<std::vector<int64_t>> getNumInputNodes() const { return mInputShapes; }
  std::vector<std::vector<int64_t>> getNumOutputNodes() const { return mOutputShapes; }
  std::vector<std::string> getInputNames() const { return mInputNames; }
  std::vector<std::string> getOutputNames() const { return mOutputNames; }
  Ort::SessionOptions* getSessionOptions();
  Ort::MemoryInfo* getMemoryInfo();
  Ort::Env* getEnv();
  int32_t getIntraOpNumThreads() const { return mIntraOpNumThreads; }
  int32_t getInterOpNumThreads() const { return mInterOpNumThreads; }

  // Setters
  void setDeviceId(int32_t id) { mDeviceId = id; }
  void setIO();
  void setActiveThreads(int threads) { mIntraOpNumThreads = threads; }
  void setIntraOpNumThreads(int threads)
  {
    if (mDeviceType == "CPU") {
      mIntraOpNumThreads = threads;
    }
  }
  void setInterOpNumThreads(int threads)
  {
    if (mDeviceType == "CPU") {
      mInterOpNumThreads = threads;
    }
  }
  void setEnv(Ort::Env*);

  // Conversion
  template <class I, class O>
  std::vector<O> v2v(std::vector<I>&, bool = true);

  // Inferencing
  template <class I, class O> // class I is the input data type, e.g. float, class O is the output data type, e.g. OrtDataType::Float16_t from O2/Common/ML/include/ML/GPUORTFloat16.h
  std::vector<O> inference(std::vector<I>&);

  template <class I, class O>
  std::vector<O> inference(std::vector<std::vector<I>>&);

  template <class I, class O>
  void inference(I*, int64_t, O*);

  template <class I, class O>
  void inference(I**, int64_t, O*);

  void release(bool = false);

 private:
  // ORT variables -> need to be hidden as pImpl
  struct OrtVariables;
  std::unique_ptr<OrtVariables> mPImplOrt;

  // Input & Output specifications of the loaded network
  std::vector<const char*> mInputNamesChar, mOutputNamesChar;
  std::vector<std::string> mInputNames, mOutputNames;
  std::vector<std::vector<int64_t>> mInputShapes, mOutputShapes, mInputShapesCopy, mOutputShapesCopy; // Input shapes
  std::vector<int64_t> mInputSizePerNode, mOutputSizePerNode;                                         // Output shapes
  int32_t mInputsTotal = 0, mOutputsTotal = 0;                                                        // Total number of inputs and outputs

  // Environment settings
  bool mInitialized = false, mDeterministicMode = false;
  std::string mModelPath, mEnvName = "", mDeviceType = "CPU", mThreadAffinity = ""; // device options should be cpu, rocm, migraphx, cuda
  int32_t mIntraOpNumThreads = 1, mInterOpNumThreads = 1, mDeviceId = -1, mEnableProfiling = 0, mLoggingLevel = 0, mAllocateDeviceMemory = 0, mEnableOptimizations = 0;

  std::string printShape(const std::vector<int64_t>&);
  std::string printShape(const std::vector<std::vector<int64_t>>&, std::vector<std::string>&);
};

} // namespace o2::ml

#endif // O2_ML_ORTINTERFACE_H

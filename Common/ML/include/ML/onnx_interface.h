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
/// \file     model.h
///
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
///
/// \brief    A general-purpose class for ONNX models
///

#ifndef COMMON_ML_ONNX_INTERFACE_H
#define COMMON_ML_ONNX_INTERFACE_H

// C++ and system includes
#if __has_include(<onnxruntime/core/session/onnxruntime_cxx_api.h>)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <thread>

// O2 includes
#include "Framework/Logger.h"

namespace o2
{

namespace ml
{

class OnnxModel
{
 public:
  OnnxModel() : mMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType, OrtMemType)) {};
  virtual ~OnnxModel() = default;

  // Inferencing
  void init(std::string, bool = false, int = 0);
  // float* inference(std::vector<Ort::Value>, int = 0);
  // float* inference(std::vector<float>, int = 0);
  template<class T> float* inference(T input, unsigned int size);
  template<class T> std::vector<float> inference_vector(T input, unsigned int size);

  // Reset session
  #if __has_include(<onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>)
    void resetSession() { mSession.reset(new Ort::Experimental::Session{*mEnv, modelPath, sessionOptions}); };
  #else
    void resetSession() { mSession.reset(new Ort::Session{*mEnv, modelPath.c_str(), sessionOptions}); };
  #endif

  // Getters & Setters
  Ort::SessionOptions* getSessionOptions() { return &sessionOptions; } // For optimizations in post
  #if __has_include(<onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>)
    std::shared_ptr<Ort::Experimental::Session> getSession() { return mSession; }
  #else
    std::shared_ptr<Ort::Session> getSession() { return mSession; }
  #endif
  std::vector<std::vector<int64_t>> getNumInputNodes() const { return mInputShapes; }
  std::vector<std::vector<int64_t>> getNumOutputNodes() const { return mOutputShapes; }
  void setActiveThreads(int);

 private:
  // Environment variables for the ONNX runtime
  std::shared_ptr<Ort::Env> mEnv = nullptr;
  std::shared_ptr<Ort::Session> mSession = nullptr; ///< ONNX session
  Ort::MemoryInfo mMemoryInfo;
  Ort::SessionOptions sessionOptions;

  // Input & Output specifications of the loaded network
  std::vector<std::string> mInputNames;
  std::vector<std::vector<int64_t>> mInputShapes;
  std::vector<std::string> mOutputNames;
  std::vector<std::vector<int64_t>> mOutputShapes;

  // Environment settings
  std::string modelPath;
  int activeThreads = 0;

  // Internal function for printing the shape of tensors
  std::string printShape(const std::vector<int64_t>&);
};

} // namespace gpu

} // namespace GPUCA_NAMESPACE

#endif // COMMON_ML_ONNX_INTERFACE_H
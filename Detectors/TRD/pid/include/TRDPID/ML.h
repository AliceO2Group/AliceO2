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

/// \file ML.h
/// \brief This file provides the base for ML policies.
/// \author Felix Schlepper

#ifndef O2_TRD_ML_H
#define O2_TRD_ML_H

#include "Rtypes.h"
#include "TRDPID/PIDBase.h"
#include "DataFormatsTRD/PID.h"
#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <array>
#include <string>

namespace o2
{
namespace trd
{

/// This is the ML Base class which defines the interface all machine learning
/// models.
class ML : public PIDBase
{
  using PIDBase::PIDBase;

 public:
  void init(o2::framework::ProcessingContext& pc) final;
  float process(const TrackTRD& trk, const o2::globaltracking::RecoContainer& input, bool isTPCTRD) const final;

 private:
  /// Return the electron likelihood.
  /// Different models have different ways to return the probability.
  virtual inline float getELikelihood(const std::vector<Ort::Value>& tensorData) const noexcept = 0;

  /// Fetch a ML model from the ccdb via its binding
  std::string fetchModelCCDB(o2::framework::ProcessingContext& pc, const char* binding) const noexcept;

  /// Prepare model input
  /// Collect track properties in vector as flat array
  std::vector<float> prepareModelInput(const TrackTRD& trkTRD, const o2::globaltracking::RecoContainer& inputTracks) const noexcept;

  /// Pretty print model shape
  std::string printShape(const std::vector<int64_t>& v) const noexcept;

  /// Get DPL name
  virtual inline std::string getName() const noexcept = 0;

  // ONNX runtime
  Ort::Env mEnv{ORT_LOGGING_LEVEL_WARNING, "TRD-PID",
                // Integrate ORT logging into Fairlogger this way we can have
                // all the nice logging while taking advantage of ORT telling us
                // what to do.
                [](void* param, OrtLoggingLevel severity, const char* category, const char* logid, const char* code_location, const char* message) {
                  LOG(warn) << "Ort " << severity << ": [" << logid << "|" << category << "|" << code_location << "]: " << message << ((intptr_t)param == 3 ? " [valid]" : " [error]");
                },
                (void*)3};                              ///< ONNX enviroment
  const OrtApi& mApi{Ort::GetApi()};                    ///< ONNX api
  std::unique_ptr<Ort::Experimental::Session> mSession; ///< ONNX session
  Ort::SessionOptions mSessionOptions;                  ///< ONNX session options

  // Input/Output
  std::vector<std::string> mInputNames;            ///< model input names
  std::vector<std::vector<int64_t>> mInputShapes;  ///< input shape
  std::vector<std::string> mOutputNames;           ///< model output names
  std::vector<std::vector<int64_t>> mOutputShapes; ///< output shape

  ClassDefOverride(ML, 1);
};

/// XGBoost Model
class XGB final : public ML
{
  using ML::ML;

 public:
  ~XGB() = default;

 private:
  /// XGBoost export is like this:
  /// (label|eprob, 1-eprob).
  inline float getELikelihood(const std::vector<Ort::Value>& tensorData) const noexcept
  {
    return tensorData[1].GetTensorData<float>()[1];
  }

  inline std::string getName() const noexcept { return "xgb"; }

  ClassDefNV(XGB, 1);
};

/// PyTorch Model
class PY final : public ML
{
  using ML::ML;

 public:
  ~PY() = default;

 private:
  inline float getELikelihood(const std::vector<Ort::Value>& tensorData) const noexcept
  {
    return tensorData[0].GetTensorData<float>()[0];
  }

  inline std::string getName() const noexcept { return "py"; }

  ClassDefNV(PY, 1);
};

} // namespace trd
} // namespace o2

#endif

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
  PIDValue process(const TrackTRD& trk, const o2::globaltracking::RecoContainer& input, bool isTPC) final;

 private:
  /// Return the electron likelihood.
  /// Different models have different ways to return the probability.
  virtual PIDValue getELikelihood(const std::vector<Ort::Value>& tensorData) const noexcept = 0;

  /// Fetch a ML model from the ccdb via its binding
  std::string fetchModelCCDB(o2::framework::ProcessingContext& pc, const char* binding) const;

  /// Calculate pid value
  template <bool isTPCTRD>
  PIDValue calculate(const TrackTRD& trkTRD, const o2::globaltracking::RecoContainer& inputTracks);

  /// Prepare model input
  /// Collect track properties in vector as flat array
  template <bool isTPCTRD>
  std::vector<float> prepareModelInput(const TrackTRD& trkTRD, const o2::globaltracking::RecoContainer& inputTracks);

  /// Pretty print model shape
  std::string printShape(const std::vector<int64_t>& v) const noexcept;

  // ONNX runtime
  Ort::Env mEnv{ORT_LOGGING_LEVEL_WARNING, "TRD-PID",
                // integrate ORT logging into Fairlogger
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

  ClassDefNV(ML, 1);
};

/// XGBoost Model
class XGB final : public ML
{
  using ML::ML;

 public:
  ~XGB() final = default;

 private:
  PIDValue getELikelihood(const std::vector<Ort::Value>& tensorData) const noexcept final;

  ClassDefNV(XGB, 1);
};

} // namespace trd
} // namespace o2

#endif

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

/// \file ML.cxx
/// \author Felix Schlepper

#include "TRDPID/ML.h"
#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/Tracklet64.h"
#include "DataFormatsTRD/HelperMethods.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/TrackParametrization.h"
#include "ReconstructionDataFormats/TrackParametrizationWithError.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "Framework/ProcessingContext.h"
#include "Framework/InputRecord.h"
#include "Framework/Logger.h"
#include "DataFormatsTRD/CalibratedTracklet.h"
#include "DetectorsBase/Propagator.h"

#include <fmt/format.h>
#include <onnxruntime/core/session/experimental_onnxruntime_cxx_api.h>
#include <boost/range.hpp>

#include <array>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <string>

using namespace o2::trd::constants;

namespace o2
{
namespace trd
{

void ML::init(o2::framework::ProcessingContext& pc)
{
  LOG(info) << "Initializating pid policy";

  // fetch the onnx model from the ccdb
  std::string model_data{fetchModelCCDB(pc, getName().c_str())};

  // disable telemtry events
  mEnv.DisableTelemetryEvents();
  LOG(info) << "Disabled Telemetry Events";

  // create session options
  mSessionOptions.SetIntraOpNumThreads(mParams.numOrtThreads);
  LOG(info) << "Set number of threads to " << mParams.numOrtThreads;

  // Sets graph optimization level
  mSessionOptions.SetGraphOptimizationLevel(static_cast<GraphOptimizationLevel>(mParams.graphOptimizationLevel));
  LOG(info) << "Set GraphOptimizationLevel to " << mParams.graphOptimizationLevel;

  // create actual session
  mSession = std::make_unique<Ort::Experimental::Session>(mEnv, reinterpret_cast<void*>(model_data.data()), model_data.size(), mSessionOptions);
  LOG(info) << "ONNX runtime session created";

  // print name/shape of inputs
  mInputNames = mSession->GetInputNames();
  mInputShapes = mSession->GetInputShapes();
  LOG(info) << "Input Node Name/Shape (" << mInputNames.size() << "):";
  for (size_t i = 0; i < mInputNames.size(); i++) {
    LOG(info) << "\t" << mInputNames[i] << " : " << printShape(mInputShapes[i]);
  }

  // print name/shape of outputs
  mOutputNames = mSession->GetOutputNames();
  mOutputShapes = mSession->GetOutputShapes();
  LOG(info) << "Output Node Name/Shape (" << mOutputNames.size() << "):";
  for (size_t i = 0; i < mOutputNames.size(); i++) {
    LOG(info) << "\t" << mOutputNames[i] << " : " << printShape(mOutputShapes[i]);
  }

  LOG(info) << "Finalization done";
}

float ML::process(const TrackTRD& trk, const o2::globaltracking::RecoContainer& inputTracks, bool isTPCTRD) const
{
  try {
    auto input = prepareModelInput(trk, inputTracks);
    // create memory mapping to vector above
    auto inputTensor = Ort::Experimental::Value::CreateTensor<float>(input.data(), input.size(),
                                                                     {static_cast<int64_t>(input.size()) / mInputShapes[0][1], mInputShapes[0][1]});
    std::vector<Ort::Value> ortTensor;
    ortTensor.push_back(std::move(inputTensor));
    auto outTensor = mSession->Run(mInputNames, ortTensor, mOutputNames);
    // every model defines its own output
    return getELikelihood(outTensor);
  } catch (const Ort::Exception& e) {
    LOG(error) << "Error running model inference, using defaults: " << e.what();
    // fill with negative elikelihood means no information
    return -1.f;
  }
}

std::string ML::fetchModelCCDB(o2::framework::ProcessingContext& pc, const char* binding) const noexcept
{
  // sanity checks
  auto ref = pc.inputs().get(binding);
  if (!ref.spec || !ref.payload) {
    throw std::runtime_error(fmt::format("A ML model with '{}' as binding does not exist!", binding));
  }

  // the model is in binary string format
  auto model_data = pc.inputs().get<std::string>(binding);
  if (model_data.empty()) {
    throw std::runtime_error(fmt::format("Did not get any data for {} model from ccdb!", binding));
  }
  return model_data;
}

std::vector<float> ML::prepareModelInput(const TrackTRD& trkTRD, const o2::globaltracking::RecoContainer& inputTracks) const noexcept
{
  // input is [charge0.0, charge0.1, charge0.2, charge1.0, ..., charge5.2, p0, ..., p5]
  std::vector<float> in(mInputShapes[0][1]);
  const auto& trackletsRaw = inputTracks.getTRDTracklets();
  auto trk = trdTRD;
  for (int iLayer = 0; iLayer < constants::NLAYER; ++iLayer) {
    int trkltId = trkTRD.getTrackletIndex(iLayer);
    if (trkltId < 0) {
      // no tracklet attached, fill with default values e.g. charge=-1.,
      in[iLayer * NCHARGES + 0] = -1.f;
      in[iLayer * NCHARGES + 1] = -1.f;
      in[iLayer * NCHARGES + 2] = -1.f;
      in[18 + iLayer] = -1.f;
      continue;
    } else {
      const auto xCalib = input.getTRDCalibratedTracklets()[trkTRD.getTrackletIndex(iLayer)].getX();
      auto bz = o2::base::Propagator::Instance()->getNominalBz();
      const auto tgl = trk.getTgl();
      const auto snp = trk.getSnpAt(o2::math_utils::sector2Angle(HelperMethods::getSector(input.getTRDTracklets()[trkIn.getTrackletIndex(iLayer)].getDetector())), xCalib, bz);
      const auto& trklt = trackletsRaw[trkltId];
      const auto [q0, q1, q2] = getCharges(trklt, iLayer, trkTRD, input, snp, tgl); // correct charges
      in[iLayer * NCHARGES + 0] = q0;
      in[iLayer * NCHARGES + 1] = q1;
      in[iLayer * NCHARGES + 2] = q2;
      in[18 + iLayer] = trk.getP();
    }
  }

  return in;
}

std::string ML::printShape(const std::vector<int64_t>& v) const noexcept
{
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++) {
    ss << v[i] << "x";
  }
  ss << v[v.size() - 1];
  return ss.str();
}

} // namespace trd
} // namespace o2

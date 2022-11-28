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

/// \file TRDPIDSpec.cxx
/// \brief This file provides the specification for calculating the pid value.
/// \author Felix Schlepper

#include "TRDWorkflow/TRDPIDSpec.h"
#include "CommonDataFormat/IRFrame.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsTRD/PID.h"
#include "TRDPID/PIDBase.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "Framework/CCDBParamSpec.h"

#include <gsl/span>

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

void TRDPIDSpec::init(o2::framework::InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mTimer.Stop();
  mTimer.Reset();
  if (!mInitDone) {
    mBase = getTRDPIDBase(mPolicy);

    mInitDone = true;
  }
}

void TRDPIDSpec::run(o2::framework::ProcessingContext& pc)
{
  // resume timer
  mTimer.Start(false);

  // get tracks
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());

  // set tracks as input then we can iterate over them
  mBase->setInput(recoData);

  // calculate pid values
  mBase->process(pc);

  // Output
  if (GTrackID::includesSource(GTrackID::Source::ITSTPCTRD, mTrkMask)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRDPID_ITSTPC", 0, Lifetime::Timeframe}, mBase->getPIDITSTPC());
    if (mUseMC) {
      // pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0, Lifetime::Timeframe}, matchLabelsITSTPC);
      // pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe}, trdLabelsITSTPC);
    }
  }
  if (GTrackID::includesSource(GTrackID::Source::TPCTRD, mTrkMask)) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRDPID_TPC", 0, Lifetime::Timeframe}, mBase->getPIDTPC());
    if (mUseMC) {
      // pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC", 0, Lifetime::Timeframe}, matchLabelsTPC);
      // pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "MCLB_TPC_TRD", 0, Lifetime::Timeframe}, trdLabelsTPC);
    }
  }

  // Stop
  mTimer.Stop();
}

void TRDPIDSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRD PID total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

o2::framework::DataProcessorSpec getTRDPIDSpec(bool useMC, PIDPolicy policy, GTrackID::mask_t src)
{
  std::shared_ptr<DataRequest> dataRequest = std::make_shared<DataRequest>();
  GTrackID::mask_t srcClu = GTrackID::getSourcesMask("TRD");
  dataRequest->requestTracks(src, false);      // Request ITS-TPC-TRD and TPC-TRD tracks
  dataRequest->requestClusters(srcClu, false); // Cluster = tracklets for trd
  auto& inputs = dataRequest->inputs;

  // Request correct PID policy data
  switch (policy) {
    case PIDPolicy::LQ1D:
      // inputs.emplace_back("LQ1D", "TRD", "LQ1D", 0, Lifetime::Condition, ccdbParamSpec("TRD/ppPID/LQ1D"));
      break;
    case PIDPolicy::LQ3D:
      // inputs.emplace_back("LQ3D", "TRD", "LQ3D", 0, Lifetime::Condition, ccdbParamSpec("TRD/ppPID/LQ3D"));
      break;
    case PIDPolicy::Test:
      inputs.emplace_back("mlTest", "TRD", "MLTEST", 0, Lifetime::Condition, ccdbParamSpec("TRD_test/pid/xgb1"));
      break;
    default:
      throw std::runtime_error("Unable to load requested PID policy data!");
  }

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              false,                             // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              false,                             // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);

  std::vector<OutputSpec> outputs;
  if (GTrackID::includesSource(GTrackID::Source::TPCTRD, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "TRDPID_TPC", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC", 0, Lifetime::Timeframe);
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_TPC_TRD", 0, Lifetime::Timeframe);
    }
  }
  if (GTrackID::includesSource(GTrackID::Source::ITSTPCTRD, src)) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "TRDPID_ITSTPC", 0, Lifetime::Timeframe);
    if (useMC) {
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0, Lifetime::Timeframe);
      outputs.emplace_back(o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0, Lifetime::Timeframe);
    }
  }

  return DataProcessorSpec{
    "TRDPID",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDPIDSpec>(useMC, src, dataRequest, ggRequest, policy)},
    Options{}};
}

} // end namespace trd
} // end namespace o2

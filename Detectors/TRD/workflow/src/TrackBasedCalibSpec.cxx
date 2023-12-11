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

/// \file   TrackBasedCalibSpec.cxx
/// \brief DPL device for creating/providing track based TRD calibration input
/// \author Ole Schmidt

#include "TRDWorkflow/TrackBasedCalibSpec.h"
#include "TRDCalibration/TrackBasedCalib.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Headers/DataHeader.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "TStopwatch.h"
#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

class TRDTrackBasedCalibDevice : public Task
{
 public:
  TRDTrackBasedCalibDevice(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool vdexb, bool gain) : mDataRequest(dr), mGGCCDBRequest(gr), mDoVdExBCalib(vdexb), mDoGainCalib(gain) {}
  ~TRDTrackBasedCalibDevice() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  bool mDoGainCalib{false};
  bool mDoVdExBCalib{false};

  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  TrackBasedCalib mCalibrator; // gather input data for calibration of vD, ExB and gain
  TStopwatch mTimer;
};

void TRDTrackBasedCalibDevice::init(InitContext& ic)
{
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mTimer.Stop();
  mTimer.Reset();
}

void TRDTrackBasedCalibDevice::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  mCalibrator.setInput(recoData);

  if (mDoVdExBCalib) {
    mCalibrator.calculateAngResHistos();
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "ANGRESHISTS", 0}, mCalibrator.getAngResHistos());
  }

  if (mDoGainCalib) {
    mCalibrator.calculateGainCalibObjs();
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "GAINCALIBHISTS", 0}, mCalibrator.getGainCalibHistos());
  }

  mCalibrator.reset();
  mTimer.Stop();
}

void TRDTrackBasedCalibDevice::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  pc.inputs().get<o2::trd::NoiseStatusMCM*>("mcmnoisemap"); // just to trigger the finaliseCCDB
  if (mDoGainCalib) {
    pc.inputs().get<o2::trd::LocalGainFactor*>("localgainfactors"); // just to trigger the finaliseCCDB
  }
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // init-once stuff
    mCalibrator.init();
  }
}

void TRDTrackBasedCalibDevice::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("TRD", "MCMNOISEMAP", 0)) {
    LOG(info) << "NoiseStatusMCM object has been updated";
    mCalibrator.setNoiseMapMCM((const o2::trd::NoiseStatusMCM*)obj);
    return;
  }
  if (matcher == ConcreteDataMatcher("TRD", "LOCALGAINFACTORS", 0)) {
    LOG(info) << "Local gain factors object has been updated";
    mCalibrator.setLocalGainFactors((const o2::trd::LocalGainFactor*)obj);
    return;
  }
}

void TRDTrackBasedCalibDevice::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRD track-based calibration total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTRDTrackBasedCalibSpec(o2::dataformats::GlobalTrackID::mask_t src, bool vdexb, bool gain)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  GTrackID::mask_t srcTrk;
  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, src)) {
    LOGF(info, "Found ITS-TPC tracks as input, loading ITS-TPC-TRD");
    srcTrk |= GTrackID::getSourcesMask("ITS-TPC-TRD");
    if (gain) {
      srcTrk |= GTrackID::getSourcesMask("ITS-TPC");
    }
  }
  if (GTrackID::includesSource(GTrackID::Source::TPC, src)) {
    LOGF(info, "Found TPC tracks as input, loading TPC-TRD");
    srcTrk |= GTrackID::getSourcesMask("TPC-TRD");
  }
  if (gain) {
    srcTrk |= GTrackID::getSourcesMask("TPC");
  }
  GTrackID::mask_t srcClu = GTrackID::getSourcesMask("TRD");         // we don't need all clusters, only TRD tracklets
  dataRequest->requestTracks(srcTrk, false);
  dataRequest->requestClusters(srcClu, false);

  auto& inputs = dataRequest->inputs;
  inputs.emplace_back("mcmnoisemap", "TRD", "MCMNOISEMAP", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/NoiseMapMCM"));
  if (gain) {
    inputs.emplace_back("localgainfactors", "TRD", "LOCALGAINFACTORS", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/LocalGainFactor"));
  }
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              false,                             // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);
  if (gain) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "GAINCALIBHISTS", 0, Lifetime::Timeframe);
  }
  if (vdexb) {
    outputs.emplace_back(o2::header::gDataOriginTRD, "ANGRESHISTS", 0, Lifetime::Timeframe);
  }
  if (!gain && !vdexb) {
    LOG(error) << "TRD track based calibration requested, but neither gain nor vD and ExB calibration enabled";
  }

  return DataProcessorSpec{
    "trd-trackbased-calib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackBasedCalibDevice>(dataRequest, ggRequest, vdexb, gain)},
    Options{}};
}

} // namespace trd
} // namespace o2

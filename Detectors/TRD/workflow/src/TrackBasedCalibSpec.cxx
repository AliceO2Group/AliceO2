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
  TRDTrackBasedCalibDevice(std::shared_ptr<DataRequest> dr) : mDataRequest(dr) {}
  ~TRDTrackBasedCalibDevice() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  std::shared_ptr<DataRequest> mDataRequest;
  TrackBasedCalib mCalibrator; // gather input data for calibration of vD, ExB and gain
  TStopwatch mTimer;
};

void TRDTrackBasedCalibDevice::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  o2::base::GeometryManager::loadGeometry();
  o2::base::Propagator::initFieldFromGRP();
  std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom()};
  mCalibrator.init();
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
  mCalibrator.calculateAngResHistos();
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "ANGRESHISTS", 0, Lifetime::Timeframe}, mCalibrator.getAngResHistos());
  mCalibrator.reset();
  mTimer.Stop();
}

void TRDTrackBasedCalibDevice::updateTimeDependentParams(ProcessingContext& pc)
{
  pc.inputs().get<o2::trd::NoiseStatusMCM*>("mcmnoisemap"); // just to trigger the finaliseCCDB
}

void TRDTrackBasedCalibDevice::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("TRD", "MCMNOISEMAP", 0)) {
    LOG(info) << "NoiseStatusMCM object has been updated";
    mCalibrator.setNoiseMapMCM((const o2::trd::NoiseStatusMCM*)obj);
  }
}

void TRDTrackBasedCalibDevice::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRD track-based calibration total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTRDTrackBasedCalibSpec(o2::dataformats::GlobalTrackID::mask_t src)
{
  std::vector<OutputSpec> outputs;
  auto dataRequest = std::make_shared<DataRequest>();

  GTrackID::mask_t srcTrk;
  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, src)) {
    LOGF(info, "Found ITS-TPC tracks as input, loading ITS-TPC-TRD");
    srcTrk |= GTrackID::getSourcesMask("ITS-TPC-TRD");
  }
  if (GTrackID::includesSource(GTrackID::Source::TPC, src)) {
    LOGF(info, "Found TPC tracks as input, loading TPC-TRD");
    srcTrk |= GTrackID::getSourcesMask("TPC-TRD");
  }
  GTrackID::mask_t srcClu = GTrackID::getSourcesMask("TRD");         // we don't need all clusters, only TRD tracklets
  dataRequest->requestTracks(srcTrk, false);
  dataRequest->requestClusters(srcClu, false);

  auto& inputs = dataRequest->inputs;
  inputs.emplace_back("mcmnoisemap", "TRD", "MCMNOISEMAP", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/NoiseMapMCM"));

  outputs.emplace_back(o2::header::gDataOriginTRD, "ANGRESHISTS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "trd-trackbased-calib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDTrackBasedCalibDevice>(dataRequest)},
    Options{}};
}

} // namespace trd
} // namespace o2

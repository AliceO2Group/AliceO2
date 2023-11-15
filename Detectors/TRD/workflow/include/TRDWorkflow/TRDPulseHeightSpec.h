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

#ifndef O2_TRD_PULSEHEIGHTSPEC_H
#define O2_TRD_PULSEHEIGHTSPEC_H

/// \file   TRDPulseHeightSpec.h
/// \brief DPL device for creating a pulse height spectrum with digits on tracks

#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "TRDCalibration/PulseHeight.h"
#include "DataFormatsTRD/PHData.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

class PuseHeightDevice : public o2::framework::Task
{
 public:
  PuseHeightDevice(std::shared_ptr<DataRequest> dr) : mDataRequest(dr) {}
  void init(o2::framework::InitContext& ic) final
  {
    mPulseHeight = std::make_unique<PulseHeight>();
    if (ic.options().get<bool>("enable-root-output")) {
      mPulseHeight->createOutputFile();
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    if (tinfo.globalRunNumberChanged) { // new run is starting
      mRunStopRequested = false;
    }
    if (mRunStopRequested) {
      std::vector<PHData> mPHValues{}; // the calibration expects data at every TF, so inject dummy
      pc.outputs().snapshot(Output{"TRD", "PULSEHEIGHT", 0, Lifetime::Timeframe}, mPHValues);
      return;
    }
    RecoContainer recoData;
    recoData.collectData(pc, *mDataRequest.get());
    auto digits = pc.inputs().get<gsl::span<o2::trd::Digit>>("digits");
    mPulseHeight->setInput(recoData, &digits);
    mPulseHeight->reset();
    mPulseHeight->process();
    pc.outputs().snapshot(Output{"TRD", "PULSEHEIGHT", 0, Lifetime::Timeframe}, mPulseHeight->getPHData());
    if (pc.transitionState() == TransitionHandlingState::Requested) {
      LOG(info) << "Run stop requested, finalizing";
      mRunStopRequested = true;
    }
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    mPulseHeight->closeOutputFile();
    if (mRunStopRequested) {
      return;
    }
  }

  void stop() final
  {
  }

 private:
  std::shared_ptr<DataRequest> mDataRequest;
  std::unique_ptr<o2::trd::PulseHeight> mPulseHeight;
  bool mRunStopRequested = false; // flag that run was stopped (and the last output is sent)
};

} // namespace trd

namespace framework
{

DataProcessorSpec getTRDPulseHeightSpec(GID::mask_t src, bool digitsFromReader)
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTRD, "PULSEHEIGHT", 0, Lifetime::Timeframe);

  bool isTPCavailable = false;
  if (GID::includesSource(GID::Source::ITSTPC, src)) {
    LOGF(debug, "Found ITS-TPC tracks as input, loading ITS-TPC-TRD");
    src |= GID::getSourcesMask("ITS-TPC-TRD");
  }
  if (GID::includesSource(GID::Source::TPC, src)) {
    LOGF(debug, "Found TPC tracks as input, loading TPC-TRD");
    src |= GID::getSourcesMask("TPC-TRD");
    isTPCavailable = true;
  }
  GID::mask_t srcClu = GID::getSourcesMask("TRD"); // we don't need all clusters, only TRD tracklets

  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestTracks(src, false);
  dataRequest->requestClusters(srcClu, false);
  dataRequest->inputs.emplace_back("digits", "TRD", "DIGITS", digitsFromReader ? 1 : 0);

  return DataProcessorSpec{
    "trd-pulseheight",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<o2::trd::PuseHeightDevice>(dataRequest)},
    Options{
      {"enable-root-output", VariantType::Bool, false, {"output PH and debug data to root file"}}}};
}
} // namespace framework
} // namespace o2

#endif // O2_TRD_PULSEHEIGHTSPEC_H

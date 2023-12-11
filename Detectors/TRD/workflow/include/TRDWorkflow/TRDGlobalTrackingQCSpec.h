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

#ifndef O2_TRD_GLOBALTRACKINGQCSPEC_H
#define O2_TRD_GLOBALTRACKINGQCSPEC_H

/// \file   TRDGlobalTrackingQCSpec.h
/// \brief Quality control for global tracking (residuals etc)
/// \author Ole Schmidt

// input TRD tracks, TRD tracklets, TRD calibrated tracklets, ITS-TPC tracks, TPC tracks
// output QC histograms

#include "Headers/DataHeader.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "DataFormatsParameters/GRPObject.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "TRDQC/Tracking.h"

#include "DetectorsBase/GRPGeomHelper.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;

namespace o2
{
namespace trd
{

class TRDGlobalTrackingQC : public Task
{
 public:
  TRDGlobalTrackingQC(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool tpcAvailable) : mDataRequest(dr), mGGCCDBRequest(gr), mTPCavailable(tpcAvailable) {}
  ~TRDGlobalTrackingQC() override = default;
  void init(InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
    if (!mTPCavailable) {
      mQC.disablePID();
    }
  }
  void run(ProcessingContext& pc) final
  {
    RecoContainer recoData;
    recoData.collectData(pc, *mDataRequest.get());
    updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
    mQC.reset();
    mQC.setInput(recoData);
    mQC.run();
    pc.outputs().snapshot(Output{"TRD", "TRACKINGQC", 0}, mQC.getTrackQC());
  }
  void endOfStream(framework::EndOfStreamContext& ec) final {}
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
      return;
    }
  }

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc)
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    static bool initOnceDone = false;
    if (!initOnceDone) { // these params need to be queried only once
      initOnceDone = true;

      mQC.init();

      // Local pad gain calibration from krypton run
      auto localGain = *(pc.inputs().get<o2::trd::LocalGainFactor*>("localgainfactors"));
      mQC.setLocalGainFactors(localGain);
    }
  }

  std::shared_ptr<DataRequest> mDataRequest;
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mTPCavailable{false};
  Tracking mQC;
};

DataProcessorSpec getTRDGlobalTrackingQCSpec(o2::dataformats::GlobalTrackID::mask_t src)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "TRACKINGQC", 0, Lifetime::Timeframe);
  auto dataRequest = std::make_shared<DataRequest>();
  bool isTPCavailable = false;

  if (GTrackID::includesSource(GTrackID::Source::ITSTPC, src)) {
    LOGF(debug, "Found ITS-TPC tracks as input, loading ITS-TPC-TRD");
    src |= GTrackID::getSourcesMask("ITS-TPC-TRD");
  }
  if (GTrackID::includesSource(GTrackID::Source::TPC, src)) {
    LOGF(debug, "Found TPC tracks as input, loading TPC-TRD");
    src |= GTrackID::getSourcesMask("TPC-TRD");
    isTPCavailable = true;
  }
  GTrackID::mask_t srcClu = GTrackID::getSourcesMask("TRD"); // we don't need all clusters, only TRD tracklets

  dataRequest->requestTracks(src, false);
  dataRequest->requestClusters(srcClu, false);
  dataRequest->inputs.emplace_back("localgainfactors", "TRD", "LOCALGAINFACTORS", 0, Lifetime::Condition, ccdbParamSpec("TRD/Calib/LocalGainFactor"));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              false,                             // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  return DataProcessorSpec{
    "trd-tracking-qc",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDGlobalTrackingQC>(dataRequest, ggRequest, isTPCavailable)},
    Options{}};
}

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTRDTrackingQCWriterSpec()
{
  return framework::MakeRootTreeWriterSpec("trd-tracking-qc-writer",
                                           "trdQC.root",
                                           "qc",
                                           BranchDefinition<std::vector<TrackQC>>{InputSpec{"trackingqc", "TRD", "TRACKINGQC"}, "trackQC"})();
};

} // namespace trd
} // namespace o2

#endif // O2_TRD_GLOBALTRACKINGQCSPEC_H

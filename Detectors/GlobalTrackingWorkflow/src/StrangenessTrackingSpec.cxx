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
/// \file StrangenessTrackingSpec.cxx
/// \brief

#include "TGeoGlobalMagField.h"
#include "Framework/ConfigParamRegistry.h"
#include "Field/MagneticField.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "StrangenessTracking/StrangenessTrackingConfigParam.h"
#include "GlobalTrackingWorkflow/StrangenessTrackingSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "Framework/CCDBParamSpec.h"
#include "DataFormatsParameters/GRPObject.h"

#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "ITStracking/IOUtils.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"

#include <fmt/format.h>
namespace o2
{
using namespace o2::framework;
namespace strangeness_tracking
{

using StrangeTrack = o2::dataformats::StrangeTrack;
using DataRequest = o2::globaltracking::DataRequest;

StrangenessTrackerSpec::StrangenessTrackerSpec(std::shared_ptr<DataRequest> dr, std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC) : mDataRequest{dr}, mGGCCDBRequest(gr), mUseMC{isMC}
{
  // no ops
}

void StrangenessTrackerSpec::init(framework::InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();

  // load propagator

  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mTracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
  mTracker.setConfigParams(&StrangenessTrackingParamConfig::Instance());
  mTracker.setupThreads(1);
  mTracker.setupFitters();

  LOG(info) << "Initialized strangeness tracker...";
}

void StrangenessTrackerSpec::run(framework::ProcessingContext& pc)
{
  mTimer.Start(false);
  LOG(debug) << "Running strangeness tracker...";

  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);

  auto geom = o2::its::GeometryTGeo::Instance();
  mTracker.loadData(recoData);
  mTracker.prepareITStracks();
  mTracker.process();
  pc.outputs().snapshot(Output{"GLO", "STRANGETRACKS", 0, Lifetime::Timeframe}, mTracker.getStrangeTrackVec());
  pc.outputs().snapshot(Output{"GLO", "CLUSUPDATES", 0, Lifetime::Timeframe}, mTracker.getClusAttachments());

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "STRANGETRACKS_MC", 0, Lifetime::Timeframe}, mTracker.getStrangeTrackLabels());
  }

  mTimer.Stop();
}

///_______________________________________
void StrangenessTrackerSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
  if (o2::base::Propagator::Instance()->getNominalBz() != mTracker.getBz()) {
    mTracker.setBz(o2::base::Propagator::Instance()->getNominalBz());
    mTracker.setupFitters();
  }
  mTracker.setMCTruthOn(mUseMC);
}

///_______________________________________
void StrangenessTrackerSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    mTracker.setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

void StrangenessTrackerSpec::endOfStream(framework::EndOfStreamContext& ec)
{
  LOGF(info, "Strangeness tracking total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getStrangenessTrackerSpec(o2::dataformats::GlobalTrackID::mask_t src, bool useMC)
{
  // ITS
  auto dataRequest = std::make_shared<DataRequest>();
  dataRequest->requestITSClusters(useMC);
  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  dataRequest->requestSecondaryVertices(useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "STRANGETRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "CLUSUPDATES", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("GLO", "STRANGETRACKS_MC", 0, Lifetime::Timeframe);
    LOG(info) << "Strangeness tracker will use MC";
  }

  return DataProcessorSpec{
    "strangeness-tracker",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<StrangenessTrackerSpec>(dataRequest, ggRequest, useMC)},
    Options{}};
}

} // namespace strangeness_tracking
} // namespace o2

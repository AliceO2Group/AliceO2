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

#include "StrangenessTrackingWorkflow/StrangenessTrackingSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/SecondaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/TOFMatcherSpec.h"
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

framework::WorkflowSpec getWorkflow(bool useMC, bool useRootInput)
{
  framework::WorkflowSpec specs;
  if (useRootInput) {
    specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(useMC, true));
    specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
    specs.emplace_back(o2::vertexing::getSecondaryVertexReaderSpec());
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(true));
    // auto src = o2::dataformats::GlobalTrackID::Source::ITSTPCTOF | o2::dataformats::GlobalTrackID::Source::ITSTPC | o2::dataformats::GlobalTrackID::Source::TPCTOF;
    // specs.emplace_back(o2::globaltracking::getTOFMatcherSpec(src, true, false, false, 0));
  }
  specs.emplace_back(getStrangenessTrackerSpec());
  return specs;
}

StrangenessTrackerSpec::StrangenessTrackerSpec(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC) : mGGCCDBRequest(gr), mIsMC{isMC}
{
  // no ops
}

void StrangenessTrackerSpec::init(framework::InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();

  // load propagator

  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);

  LOG(info) << "Initialized strangeness tracker...";
}

void StrangenessTrackerSpec::run(framework::ProcessingContext& pc)
{
  mTimer.Start(false);
  LOG(info) << "Running strangeness tracker...";
  updateTimeDependentParams(pc);
  // ITS
  auto ITSclus = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  auto ITSpatt = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto ITStracks = pc.inputs().get<gsl::span<o2::its::TrackITS>>("ITSTrack");
  auto ROFsInput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  auto ITSTrackClusIdx = pc.inputs().get<gsl::span<int>>("trackITSClIdx");

  // V0
  auto v0Vec = pc.inputs().get<gsl::span<o2::dataformats::V0>>("v0s");
  auto cascadeVec = pc.inputs().get<gsl::span<o2::dataformats::Cascade>>("cascs");
  auto tpcITSTracks = pc.inputs().get<gsl::span<o2::dataformats::TrackTPCITS>>("trackTPCITS");

  // Monte Carlo
  auto labITSTPC = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSTPCMCTR");
  // auto labTPCTOF = pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_TPC_MCTR");
  // auto labITSTPCTOF = pc.inputs().get<gsl::span<o2::MCCompLabel>>("clsTOF_GLO_MCTR");
  auto labITS = pc.inputs().get<gsl::span<o2::MCCompLabel>>("trackITSMCTR");
  LOGF(debug, "ITSclus: %d \nITSpatt: %d \nITStracks: %d \nROFsInput: %d \nITSTrackClusIdx: %d \nTPCITStracks: %d \nv0s: %d \nlabITSTPC: %d\nlabITS: %d",
       ITSclus.size(),
       ITSpatt.size(),
       ITStracks.size(),
       ROFsInput.size(),
       ITSTrackClusIdx.size(),
       tpcITSTracks.size(),
       v0Vec.size(),
       cascadeVec.size(),
       labITSTPC.size(),
       //  labTPCTOF.size(),
       //  labITSTPCTOF.size(),
       labITS.size());
  //  \nlabTPCTOF: %d\nlabITSTPCTOF: %d

  mTracker.setCorrType(o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT);
  LOG(debug) << "Bz: " << o2::base::Propagator::Instance()->getNominalBz();
  mTracker.setBz(o2::base::Propagator::Instance()->getNominalBz());

  auto pattIt = ITSpatt.begin();
  std::vector<ITSCluster> ITSClustersArray;
  ITSClustersArray.reserve(ITSclus.size());
  o2::its::ioutils::convertCompactClusters(ITSclus, pattIt, ITSClustersArray, mDict);
  auto geom = o2::its::GeometryTGeo::Instance();

  mTracker.loadData(ITStracks, ITSClustersArray, ITSTrackClusIdx, v0Vec, cascadeVec, geom);
  mTracker.initialise();
  mTracker.process();

  pc.outputs().snapshot(Output{"STK", "STRTRACKS", 0, Lifetime::Timeframe}, mTracker.getStrangeTrackVec());
  pc.outputs().snapshot(Output{"STK", "CLUSUPDATES", 0, Lifetime::Timeframe}, mTracker.getClusAttachments());

  mTimer.Stop();
}

///_______________________________________
void StrangenessTrackerSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot, o2::math_utils::TransformType::T2G));
  }
}

///_______________________________________
void StrangenessTrackerSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
}

void StrangenessTrackerSpec::endOfStream(framework::EndOfStreamContext& ec)
{
  LOGF(info, "Strangeness tracking total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getStrangenessTrackerSpec()
{
  std::vector<InputSpec> inputs;

  // ITS
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("trackITSClIdx", "ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  inputs.emplace_back("ITSTrack", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("cldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);

  // V0
  inputs.emplace_back("v0s", "GLO", "V0S", 0, Lifetime::Timeframe);                // found V0s
  inputs.emplace_back("v02pvrf", "GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);    // prim.vertex -> V0s refs
  inputs.emplace_back("cascs", "GLO", "CASCS", 0, Lifetime::Timeframe);            // found Cascades
  inputs.emplace_back("cas2pvrf", "GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe); // prim.vertex -> Cascades refs

  // TPC-ITS
  inputs.emplace_back("trackTPCITS", "GLO", "TPCITS", 0, Lifetime::Timeframe);

  // TPC-TOF
  // inputs.emplace_back("matchTPCTOF", "TOF", "MTC_TPC", 0, Lifetime::Timeframe); // Matching input type manually set to 0

  // Monte Carlo
  inputs.emplace_back("trackITSTPCMCTR", "GLO", "TPCITS_MC", 0, Lifetime::Timeframe); // MC truth
  // inputs.emplace_back("clsTOF_GLO_MCTR", "TOF", "MCMTC_ITSTPC", 0, Lifetime::Timeframe); // MC truth
  inputs.emplace_back("trackITSMCTR", "ITS", "TRACKSMCTR", 0, Lifetime::Timeframe); // MC truth
  // inputs.emplace_back("clsTOF_TPC_MCTR", "TOF", "MCMTC_TPC", 0, Lifetime::Timeframe);    // MC truth, // Matching input type manually set to 0

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("STK", "STRTRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("STK", "CLUSUPDATES", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "strangeness-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<StrangenessTrackerSpec>(ggRequest, false)},
    Options{}};
}

} // namespace strangeness_tracking
} // namespace o2
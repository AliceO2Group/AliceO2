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

/// @file   CookedTrackerSpec.cxx

#include <vector>

#include "TGeoGlobalMagField.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "ITSWorkflow/CookedTrackerSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include <DataFormatsITSMFT/PhysTrigger.h>
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DataFormatsTRD/TriggerRecord.h"

#include "Field/MagneticField.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "ITSBase/GeometryTGeo.h"
#include "CommonDataFormat/IRFrame.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/IOUtils.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "CommonUtils/StringUtils.h"

#include "ITSReconstruction/FastMultEstConfig.h"
#include "ITSReconstruction/FastMultEst.h"
#include "ITSMFTReconstruction/ClustererParam.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

CookedTrackerDPL::CookedTrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool useMC, int trgType, const std::string& trMode) : mGGCCDBRequest(gr), mUseMC(useMC), mUseTriggers{trgType}, mMode(trMode)
{
  mVertexerTraitsPtr = std::make_unique<VertexerTraits>();
  mVertexerPtr = std::make_unique<Vertexer>(mVertexerTraitsPtr.get());
}

void CookedTrackerDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  auto nthreads = ic.options().get<int>("nthreads");
  mTracker.setNumberOfThreads(nthreads);
}

void CookedTrackerDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  updateTimeDependentParams(pc);
  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  gsl::span<const o2::itsmft::PhysTrigger> physTriggers;
  std::vector<o2::itsmft::PhysTrigger> fromTRD;
  if (mUseTriggers == 2) { // use TRD triggers
    o2::InteractionRecord ir{0, pc.services().get<o2::framework::TimingInfo>().firstTForbit};
    auto trdTriggers = pc.inputs().get<gsl::span<o2::trd::TriggerRecord>>("phystrig");
    for (const auto& trig : trdTriggers) {
      if (trig.getBCData() >= ir && trig.getNumberOfTracklets()) {
        ir = trig.getBCData();
        fromTRD.emplace_back(o2::itsmft::PhysTrigger{ir, 0});
      }
    }
    physTriggers = gsl::span<const o2::itsmft::PhysTrigger>(fromTRD.data(), fromTRD.size());
  } else if (mUseTriggers == 1) { // use Phys triggers from ITS stream
    physTriggers = pc.inputs().get<gsl::span<o2::itsmft::PhysTrigger>>("phystrig");
  }

  auto& rofs = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "ITSTrackROF", 0, Lifetime::Timeframe}, rofsinput.begin(), rofsinput.end());

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  gsl::span<itsmft::MC2ROFRecord const> mc2rofs;
  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    // get the array as read-onlt span, a snapshot is send forward
    mc2rofs = pc.inputs().get<gsl::span<itsmft::MC2ROFRecord>>("MC2ROframes");
  }
  TimeFrame mTimeFrame;

  LOG(info) << "ITSCookedTracker pulled " << compClusters.size() << " clusters, in " << rofs.size() << " RO frames";

  std::vector<o2::MCCompLabel> trackLabels;
  if (mUseMC) {
    mTracker.setMCTruthContainers(labels.get(), &trackLabels);
  }

  o2::its::ROframe event(0, 7);
  mVertexerPtr->adoptTimeFrame(mTimeFrame);

  auto& vertROFvec = pc.outputs().make<std::vector<o2::itsmft::ROFRecord>>(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe});
  auto& vertices = pc.outputs().make<std::vector<Vertex>>(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe});
  auto& tracks = pc.outputs().make<std::vector<o2::its::TrackITS>>(Output{"ITS", "TRACKS", 0, Lifetime::Timeframe});
  auto& clusIdx = pc.outputs().make<std::vector<int>>(Output{"ITS", "TRACKCLSID", 0, Lifetime::Timeframe});
  auto& irFrames = pc.outputs().make<std::vector<o2::dataformats::IRFrame>>(Output{"ITS", "IRFRAMES", 0, Lifetime::Timeframe});

  const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance(); // RS: this should come from CCDB
  int nBCPerTF = mTracker.getContinuousMode() ? alpParams.roFrameLengthInBC : alpParams.roFrameLengthTrig;

  gsl::span<const unsigned char>::iterator pattIt_timeframe = patterns.begin();
  gsl::span<const unsigned char>::iterator pattIt_tracker = patterns.begin();
  gsl::span<itsmft::ROFRecord> rofspan(rofs);
  mTimeFrame.loadROFrameData(rofspan, compClusters, pattIt_timeframe, mDict, labels.get());

  const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  FastMultEst multEst;                                     // mult estimator
  std::vector<bool> processingMask;
  int cutVertexMult{0}, cutRandomMult = int(rofsinput.size()) - multEst.selectROFs(rofsinput, compClusters, physTriggers, processingMask);

  // auto processingMask_ephemeral = processingMask;
  mTimeFrame.setMultiplicityCutMask(processingMask);
  float vertexerElapsedTime;
  if (mRunVertexer) {
    vertexerElapsedTime = mVertexerPtr->clustersToVertices(false, [&](std::string s) { LOG(info) << s; });
  }
  LOG(info) << fmt::format(" - Vertex seeding total elapsed time: {} ms in {} ROFs", vertexerElapsedTime, rofspan.size());
  for (size_t iRof{0}; iRof < rofspan.size(); ++iRof) {
    auto& rof = rofspan[iRof];

    auto& vtxROF = vertROFvec.emplace_back(rof); // register entry and number of vertices in the
    vtxROF.setFirstEntry(vertices.size());
    vtxROF.setNEntries(0);
    if (!processingMask[iRof]) {
      rof.setFirstEntry(tracks.size());
      rof.setNEntries(0);
      continue;
    }

    std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>> vtxVecLoc;
    for (auto& v : mTimeFrame.getPrimaryVertices(iRof)) {
      vtxVecLoc.push_back(v);
    }

    if (multEstConf.isVtxMultCutRequested()) { // cut was requested
      std::vector<o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>> vtxVecSel;
      vtxVecSel.swap(vtxVecLoc);
      int nv = vtxVecSel.size(), nrej = 0;
      for (const auto& vtx : vtxVecSel) {
        if (!multEstConf.isPassingVtxMultCut(vtx.getNContributors())) {
          LOG(info) << "Found vertex mult. " << vtx.getNContributors() << " is outside of requested range " << multEstConf.cutMultVtxLow << " : " << multEstConf.cutMultVtxHigh << " | ROF " << rof.getBCData();
          nrej++;
          continue; // skip vertex of unwanted multiplicity
        }
        vtxVecLoc.push_back(vtx);
      }
      if (nv && (nrej == nv)) { // all vertices were rejected
        cutVertexMult++;
        processingMask[iRof] = false;
      }
    }
    if (vtxVecLoc.empty()) {
      if (multEstConf.cutMultVtxLow < 1) { // do blind search only if there is no cut on the low mult vertices
        vtxVecLoc.emplace_back();
      } else {
        rof.setFirstEntry(tracks.size());
        rof.setNEntries(0);
        continue;
      }
    } else { // save vertices
      vtxROF.setNEntries(vtxVecLoc.size());
      for (const auto& vtx : vtxVecLoc) {
        vertices.push_back(vtx);
      }
    }

    mTracker.setVertices(vtxVecLoc);
    mTracker.process(compClusters, pattIt_tracker, mDict, tracks, clusIdx, rof);
    if (processingMask[iRof]) {
      irFrames.emplace_back(rof.getBCData(), rof.getBCData() + nBCPerTF - 1).info = tracks.size();
    }
  }
  LOGP(info, " - rejected {}/{} ROFs: random/mult.sel:{}, vtx.sel:{}", cutRandomMult + cutVertexMult, rofspan.size(), cutRandomMult, cutVertexMult);
  LOG(info) << "ITSCookedTracker pushed " << tracks.size() << " tracks and " << vertices.size() << " vertices";

  if (mUseMC) {
    pc.outputs().snapshot(Output{"ITS", "TRACKSMCTR", 0, Lifetime::Timeframe}, trackLabels);
    pc.outputs().snapshot(Output{"ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
  }
  mTimer.Stop();
}

void CookedTrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "ITS Cooked-Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

///_______________________________________
void CookedTrackerDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    pc.inputs().get<o2::itsmft::TopologyDictionary*>("cldict"); // just to trigger the finaliseCCDB
    pc.inputs().get<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>*>("alppar");

    mVertexerPtr->getGlobalConfiguration();
    o2::its::GeometryTGeo* geom = o2::its::GeometryTGeo::Instance();
    geom->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                   o2::math_utils::TransformType::T2G));
    mTracker.setGeometry(geom);
    mTracker.setConfigParams();
    LOG(info) << "Tracking mode " << mMode;
    if (mMode == "cosmics") {
      LOG(info) << "Setting cosmics parameters...";
      mTracker.setParametersCosmics();
      mRunVertexer = false;
    }
    mTracker.setBz(o2::base::Propagator::Instance()->getNominalBz());
    bool continuous = o2::base::GRPGeomHelper::instance().getGRPECS()->isDetContinuousReadOut(o2::detectors::DetID::ITS);
    LOG(info) << "ITSCookedTracker RO: continuous=" << continuous;
    mTracker.setContinuousMode(continuous);
  }
}

///_______________________________________
void CookedTrackerDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (matcher == ConcreteDataMatcher("ITS", "CLUSDICT", 0)) {
    LOG(info) << "cluster dictionary updated";
    setClusterDictionary((const o2::itsmft::TopologyDictionary*)obj);
    return;
  }
  // Note: strictly speaking, for Configurable params we don't need finaliseCCDB check, the singletons are updated at the CCDB fetcher level
  if (matcher == ConcreteDataMatcher("ITS", "ALPIDEPARAM", 0)) {
    LOG(info) << "Alpide param updated";
    const auto& par = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
    par.printKeyValues();
    return;
  }
}

DataProcessorSpec getCookedTrackerSpec(bool useMC, int trgType, const std::string& trMode)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  inputs.emplace_back("cldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  inputs.emplace_back("alppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));
  if (trgType == 1) {
    inputs.emplace_back("phystrig", "ITS", "PHYSTRIG", 0, Lifetime::Timeframe);
  } else if (trgType == 2) {
    inputs.emplace_back("phystrig", "TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICESROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "IRFRAMES", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
  }
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              inputs,
                                                              true);
  return DataProcessorSpec{
    "its-cooked-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CookedTrackerDPL>(ggRequest, useMC, trgType, trMode)},
    Options{{"nthreads", VariantType::Int, 1, {"Number of threads"}}}};
}

} // namespace its
} // namespace o2

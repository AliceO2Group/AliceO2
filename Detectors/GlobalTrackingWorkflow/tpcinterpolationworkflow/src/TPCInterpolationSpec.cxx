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

/// @file  TPCInterpolationSpec.cxx

#include <vector>
#include <unordered_map>

#include "DataFormatsITS/TrackITS.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DetectorsBase/GeometryManager.h"
#include "DetectorsBase/Propagator.h"
#include "TPCInterpolationWorkflow/TPCInterpolationSpec.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SpacePoints/SpacePointsCalibParam.h"
#include "SpacePoints/SpacePointsCalibConfParam.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"

using namespace o2::framework;
using namespace o2::globaltracking;
using GTrackID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2
{
namespace tpc
{

void TPCInterpolationDPL::init(InitContext& ic)
{
  //-------- init geometry and field --------//
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mSlotLength = ic.options().get<uint32_t>("sec-per-slot");
}

void TPCInterpolationDPL::updateTimeDependentParams(ProcessingContext& pc)
{
  o2::base::GRPGeomHelper::instance().checkUpdates(pc);
  o2::tpc::VDriftHelper::extractCCDBInputs(pc);
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
    // other init-once stuff
    mInterpolation.init();
    int nTfs = mSlotLength / (o2::base::GRPGeomHelper::getNHBFPerTF() * o2::constants::lhc::LHCOrbitMUS * 1e-6);
    bool limitTracks = (SpacePointsCalibConfParam::Instance().maxTracksPerCalibSlot < 0) ? true : false;
    int nTracksPerTfMax = (nTfs > 0 && !limitTracks) ? SpacePointsCalibConfParam::Instance().maxTracksPerCalibSlot / nTfs : -1;
    if (nTracksPerTfMax > 0) {
      LOGP(info, "We will stop processing tracks after validating {} tracks per TF", nTracksPerTfMax);
    } else if (nTracksPerTfMax < 0) {
      LOG(info) << "The number of processed tracks per TF is not limited";
    } else {
      LOG(error) << "No tracks will be processed. maxTracksPerCalibSlot must be greater than slot length in TFs";
    }
    mInterpolation.setMaxTracksPerTF(nTracksPerTfMax);
  }
  // we may have other params which need to be queried regularly
  if (mTPCVDriftHelper.isUpdated()) {
    LOGP(info, "Updating TPC fast transform map with new VDrift factor of {} wrt reference {} from source {}",
         mTPCVDriftHelper.getVDriftObject().corrFact, mTPCVDriftHelper.getVDriftObject().refVDrift, mTPCVDriftHelper.getSourceName());
    mInterpolation.setTPCVDrift(mTPCVDriftHelper.getVDriftObject());
    mTPCVDriftHelper.acknowledgeUpdate();
  }
}

void TPCInterpolationDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj)) {
    return;
  }
  if (mTPCVDriftHelper.accountCCDBInputs(matcher, obj)) {
    return;
  }
}

void TPCInterpolationDPL::run(ProcessingContext& pc)
{
  LOG(info) << "TPC Interpolation Workflow initialized. Start processing...";
  mTimer.Start(false);
  RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest.get());
  updateTimeDependentParams(pc);
  const auto& param = SpacePointsCalibConfParam::Instance();

  // load the input tracks
  std::vector<o2::globaltracking::RecoContainer::GlobalIDSet> gidTables;
  std::vector<o2::track::TrackParCov> seeds;
  std::vector<float> trkTimes;
  std::vector<GTrackID> gids;
  std::unordered_map<int, int> trkCounters;
  // make sure the map has entries for every possible track input type
  trkCounters.insert(std::make_pair<int, int>(GTrackID::Source::ITSTPCTRDTOF, 0));
  trkCounters.insert(std::make_pair<int, int>(GTrackID::Source::ITSTPCTRD, 0));
  trkCounters.insert(std::make_pair<int, int>(GTrackID::Source::ITSTPCTOF, 0));
  trkCounters.insert(std::make_pair<int, int>(GTrackID::Source::ITSTPC, 0));

  bool processITSTPConly = mProcessITSTPConly; // so that the flag can be used inside the lambda
  // the creator goes from most complete track (ITS-TPC-TRD-TOF) to least complete one (ITS-TPC)
  auto creator = [&gidTables, &seeds, &trkTimes, &recoData, &processITSTPConly, &gids, &param, &trkCounters](auto& _tr, GTrackID _origID, float t0, float tErr) {
    if constexpr (std::is_base_of_v<o2::track::TrackParCov, std::decay_t<decltype(_tr)>>) {
      bool trackGood = true;
      bool hasOuterPoint = false;
      auto gidTable = recoData.getSingleDetectorRefs(_origID);
      if (!gidTable[GTrackID::ITS].isIndexSet() || !gidTable[GTrackID::TPC].isIndexSet()) {
        // ITS and TPC track is always needed. At this stage ITS afterburner tracks are also rejected
        return true;
      }
      if (gidTable[GTrackID::TRD].isIndexSet() || gidTable[GTrackID::TOF].isIndexSet()) {
        hasOuterPoint = true;
      }
      const auto itstpcTrk = &recoData.getTPCITSTrack(gidTable[GTrackID::ITSTPC]);
      const auto itsTrk = &recoData.getITSTrack(gidTable[GTrackID::ITS]);
      const auto tpcTrk = &recoData.getTPCTrack(gidTable[GTrackID::TPC]);
      // apply track quality cuts
      if (itsTrk->getChi2() / itsTrk->getNumberOfClusters() > param.maxITSChi2 || tpcTrk->getChi2() / tpcTrk->getNClusterReferences() > param.maxTPCChi2) {
        // reduced chi2 cut is the same for all track types
        trackGood = false;
      }
      if (!hasOuterPoint) {
        // ITS-TPC track (does not have outer points in TRD or TOF)
        if (!processITSTPConly) {
          return true;
        }
        if (itsTrk->getNumberOfClusters() < param.minITSNClsNoOuterPoint || tpcTrk->getNClusterReferences() < param.minTPCNClsNoOuterPoint) {
          trackGood = false;
        }
      } else {
        if (itsTrk->getNumberOfClusters() < param.minITSNCls || tpcTrk->getNClusterReferences() < param.minTPCNCls) {
          trackGood = false;
        }
      }
      if (trackGood) {
        trkTimes.push_back(t0);
        seeds.emplace_back(itsTrk->getParamOut()); // FIXME: should this not be a refit of the ITS track?
        gidTables.emplace_back(gidTable);
        gids.push_back(_origID);
        trkCounters[_origID.getSource()] += 1;
      }
      return true;
    } else {
      return false;
    }
  };
  recoData.createTracksVariadic(creator); // create track sample considered for interpolation
  LOG(info) << "Created " << seeds.size() << " seeds.";

  if (mUseMC) {
    // possibly MC labels will be used to check filtering procedure performance before interpolation
    // not yet implemented
  }

  mInterpolation.process(recoData, gids, gidTables, seeds, trkTimes, trkCounters);
  mTimer.Stop();
  LOGF(info, "TPC interpolation timing: Cpu: %.3e Real: %.3e s", mTimer.CpuTime(), mTimer.RealTime());

  if (param.writeUnfiltered) {
    // these are the residuals and tracks before outlier rejection; they are not used in production
    pc.outputs().snapshot(Output{"GLO", "TPCINT_RES", 0, Lifetime::Timeframe}, mInterpolation.getClusterResidualsUnfiltered());
    if (mSendTrackData) {
      pc.outputs().snapshot(Output{"GLO", "TPCINT_TRK", 0, Lifetime::Timeframe}, mInterpolation.getReferenceTracksUnfiltered());
    }
  }
  pc.outputs().snapshot(Output{"GLO", "UNBINNEDRES", 0, Lifetime::Timeframe}, mInterpolation.getClusterResiduals());
  pc.outputs().snapshot(Output{"GLO", "TRKREFS", 0, Lifetime::Timeframe}, mInterpolation.getTrackDataCompact());
  if (mSendTrackData) {
    pc.outputs().snapshot(Output{"GLO", "TRKDATA", 0, Lifetime::Timeframe}, mInterpolation.getReferenceTracks());
  }

  mInterpolation.reset();
}

void TPCInterpolationDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TPC residuals extraction total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTPCInterpolationSpec(GTrackID::mask_t src, bool useMC, bool processITSTPConly, bool sendTrackData)
{
  auto dataRequest = std::make_shared<DataRequest>();
  std::vector<OutputSpec> outputs;

  if (useMC) {
    LOG(fatal) << "MC usage must be disabled for this workflow, since it is not yet implemented";
  }

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestClusters(src, useMC);

  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                              true,                              // GRPECS=true
                                                              false,                             // GRPLHCIF
                                                              true,                              // GRPMagField
                                                              true,                              // askMatLUT
                                                              o2::base::GRPGeomRequest::Aligned, // geometry
                                                              dataRequest->inputs,
                                                              true);
  o2::tpc::VDriftHelper::requestCCDBInputs(dataRequest->inputs);
  if (SpacePointsCalibConfParam::Instance().writeUnfiltered) {
    outputs.emplace_back("GLO", "TPCINT_TRK", 0, Lifetime::Timeframe);
    if (sendTrackData) {
      outputs.emplace_back("GLO", "TPCINT_RES", 0, Lifetime::Timeframe);
    }
  }
  outputs.emplace_back("GLO", "UNBINNEDRES", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TRKREFS", 0, Lifetime::Timeframe);
  if (sendTrackData) {
    outputs.emplace_back("GLO", "TRKDATA", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-track-interpolation",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TPCInterpolationDPL>(dataRequest, ggRequest, useMC, processITSTPConly, sendTrackData)},
    Options{
      {"sec-per-slot", VariantType::UInt32, 600u, {"number of seconds per calibration time slot (put 0 for infinite slot length)"}}}};
}

} // namespace tpc
} // namespace o2

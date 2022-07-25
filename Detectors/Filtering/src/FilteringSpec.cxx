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

/// @file   FilteringSpec.cxx

#include "FilteringSpec.h"
#include "DataFormatsFT0/RecPoints.h"
#include "DataFormatsFDD/RecPoint.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsZDC/ZDCEnergy.h"
#include "DataFormatsZDC/ZDCTDCData.h"
#include "CommonUtils/NameConf.h"
#include "MathUtils/Utils.h"
#include "DetectorsBase/GeometryManager.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonConstants/PhysicsConstants.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/CCDBParamSpec.h"
#include "FDDBase/Constants.h"
#include "FT0Base/Geometry.h"
#include "FV0Base/Geometry.h"
#include "GlobalTracking/MatchTOF.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "MCHTracking/TrackExtrap.h"
#include "MCHTracking/TrackParam.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "DetectorsVertexing/PVertexerParams.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "ReconstructionDataFormats/GlobalFwdTrack.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "SimulationDataFormat/MCEventLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCUtils.h"
#include "ZDCBase/Constants.h"
#include "TPCBase/ParameterElectronics.h"
#include "GPUTPCGMMergedTrackHit.h"
#include "TOFBase/Utils.h"
#include "TMath.h"
#include "MathUtils/Utils.h"
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;
using PVertex = o2::dataformats::PrimaryVertex;
using GIndex = o2::dataformats::VtxTrackIndex;
using DataRequest = o2::globaltracking::DataRequest;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

namespace o2::filtering
{

void FilteringSpec::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  o2::globaltracking::RecoContainer recoData;
  recoData.collectData(pc, *mDataRequest);
  updateTimeDependentParams(pc); // Make sure this is called after recoData.collectData, which may load some conditions
  mStartIR = recoData.startIR;

  auto primVer2TRefs = recoData.getPrimaryVertexMatchedTrackRefs();
  /*
  // examples of accessing different reco data objects

  auto primVertices = recoData.getPrimaryVertices();
  auto primVerLabels = recoData.getPrimaryVertexMCLabels();

  auto fddChData = recoData.getFDDChannelsData();
  auto fddRecPoints = recoData.getFDDRecPoints();
  auto ft0ChData = recoData.getFT0ChannelsData();
  auto ft0RecPoints = recoData.getFT0RecPoints();
  auto fv0ChData = recoData.getFV0ChannelsData();
  auto fv0RecPoints = recoData.getFV0RecPoints();

  auto zdcEnergies = recoData.getZDCEnergy();
  auto zdcBCRecData = recoData.getZDCBCRecData();
  auto zdcTDCData = recoData.getZDCTDCData();

  // get calo information
  auto caloEMCCells = recoData.getEMCALCells();
  auto caloEMCCellsTRGR = recoData.getEMCALTriggers();

  auto ctpDigits = recoData.getCTPDigits();
  // std::unique_ptr<o2::steer::MCKinematicsReader> mcReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
  */

  // process tracks associated with vertices, note that the last entry corresponds to orphan tracks (no vertex)
  for (const auto& trackRef : primVer2TRefs) {
    processTracksOfVertex(trackRef, recoData);
  }

  if (mNeedToSave) {
    fillData(recoData);
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    mFTF.header.run = tinfo.runNumber;
    mFTF.header.firstTForbit = tinfo.firstTForbit;
    mFTF.header.creationTime = tinfo.creation;

    pc.outputs().snapshot(Output{"GLO", "FILTERED_RECO_TF", 0}, mFTF);
    clear(); // clear caches, safe after the snapshot
  }

  mTimer.Stop();
}

void FilteringSpec::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ITS", "CLUSTERDICT", 0)) {
    LOG(info) << "ITS cluster dictionary updated";
    mDictITS = (o2::itsmft::TopologyDictionary*)obj;
    return;
  }
}

void FilteringSpec::init(InitContext& ic)
{
}

void FilteringSpec::clear()
{
  mNeedToSave = false;
  mFTF.clear(); // we can clear immidiately after the snapshot
  mGIDToTableID.clear();
  mITSTrackIDCache.clear();
  mITSClusterIDCache.clear();
}

void FilteringSpec::fillData(const o2::globaltracking::RecoContainer& recoData)
{
  // actual data filling

  // ITS tracks
  if (!mITSTrackIDCache.empty()) {
    const auto tracksOrig = recoData.getITSTracks();
    const auto tracksOrigLbl = recoData.getITSTracksMCLabels();
    const auto clusRefOrig = recoData.getITSTracksClusterRefs();
    const auto rofsOrig = recoData.getITSTracksROFRecords();
    auto trIt = mITSTrackIDCache.begin(); // tracks sorted in their ID
    for (unsigned irof = 0; irof < rofsOrig.size() && trIt != mITSTrackIDCache.end(); irof++) {
      const auto& rofOrig = rofsOrig[irof];
      int startID = rofOrig.getFirstEntry(), endID = startID + rofOrig.getNEntries();
      if (trIt->first >= endID) {
        continue; // nothing from this ROF is selected
      }
      auto& rofSave = mFTF.ITSTrackROFs.emplace_back(rofOrig);
      rofSave.setFirstEntry(mFTF.ITSTracks.size());
      while (trIt != mITSTrackIDCache.end() && trIt->first < endID) {
        trIt->second = mFTF.ITSTracks.size(); // this for the further bookkeeping?
        const auto& trOr = tracksOrig[trIt->first];
        auto& trSave = mFTF.ITSTracks.emplace_back(trOr);
        trSave.setFirstClusterEntry(mFTF.ITSClusterIndices.size()); // N cluster entries is set correctly at creation
        for (int i = 0; i < trOr.getNClusters(); i++) {
          int clID = trOr.getClusterEntry(i);
          mFTF.ITSClusterIndices.push_back(clID); // later will be remapped to actually stored cluster indices
          mITSClusterIDCache[clID] = 0;           // flag used cluster
        }
        if (mUseMC) {
          mFTF.ITSTrackMCTruth.push_back(tracksOrigLbl[trIt->first]);
        }
        trIt++;
      }
      rofSave.setNEntries(mFTF.ITSTracks.size() - rofSave.getFirstEntry());
    }
  }
  // ITS clusters info for selected tracks
  if (!mITSClusterIDCache.empty()) {
    const auto clusOrig = recoData.getITSClusters();
    const auto pattOrig = recoData.getITSClustersPatterns();
    const auto rofsOrig = recoData.getITSClustersROFRecords();
    auto clIt = mITSClusterIDCache.begin(); // clusters sorted in their ID
    auto pattItOrig = pattOrig.begin(), pattItOrigPrev = pattItOrig;
    for (unsigned irof = 0; irof < rofsOrig.size() && clIt != mITSClusterIDCache.end(); irof++) {
      const auto& rofOrig = rofsOrig[irof];
      int startID = rofOrig.getFirstEntry(), endID = startID + rofOrig.getNEntries();
      if (clIt->first >= endID) {
        continue; // nothing from this ROF is selected
      }
      auto& rofSave = mFTF.ITSClusterROFs.emplace_back(rofOrig);
      rofSave.setFirstEntry(mFTF.ITSTracks.size());
      while (clIt != mITSClusterIDCache.end() && clIt->first < endID) {
        clIt->second = mFTF.ITSClusters.size(); // new index of the stored cluster
        const auto& clOr = clusOrig[clIt->first];
        auto& clSave = mFTF.ITSClusters.emplace_back(clOr);
        // cluster pattern
        auto pattID = clOr.getPatternID();
        o2::itsmft::ClusterPattern patt;
        if (pattID == o2::itsmft::CompCluster::InvalidPatternID || mDictITS->isGroup(pattID)) {
          patt.acquirePattern(pattItOrig);
        }
        while (pattItOrigPrev != pattItOrig) { // the difference, if any, is the explicitly stored pattern
          mFTF.ITSClusterPatterns.push_back(*pattItOrigPrev);
          pattItOrigPrev++;
        }
        clIt++;
      }
      rofSave.setNEntries(mFTF.ITSClusters.size() - rofSave.getFirstEntry());
    }
  }
  // now remap cluster indices to stored values
  for (auto& clID : mFTF.ITSClusterIndices) {
    clID = mITSClusterIDCache[clID];
  }
}

void FilteringSpec::processTracksOfVertex(const o2::dataformats::VtxTrackRef& trackRef, const o2::globaltracking::RecoContainer& recoData)
{
  auto GIndices = recoData.getPrimaryVertexMatchedTracks(); // Global IDs of all tracks
  int vtxID = trackRef.getVtxID();                          // -1 means that we are processing orphan tracks

  for (int src = GIndex::NSources; src--;) { // loop over all possible types of tracks
    int start = trackRef.getFirstEntryOfSource(src);
    int end = start + trackRef.getEntriesOfSource(src);
    for (int ti = start; ti < end; ti++) {
      auto& trackIndex = GIndices[ti];
      if (GIndex::includesSource(src, mInputSources)) { // should we consider this source

        if (trackIndex.isAmbiguous()) { // this trak was matched to multiple vertices
          const auto res = mGIDToTableID.find(trackIndex);
          if (res != mGIDToTableID.end()) { // and was already processed
            if (res->second < 0) {          // was selected, just register its vertex
              // registerVertex // FIXME: will be done once processing of all track types is complete
            }
            continue;
          }
        }
        // here we select barrel tracks only (they must have TPC or ITS contributions)
        if (!trackIndex.includesDet(DetID::ITS) && !trackIndex.includesDet(DetID::TPC)) {
          continue;
        }

        int selRes = processBarrelTrack(trackIndex, recoData); // was track selected?
        if (trackIndex.isAmbiguous()) {                        // remember decision on this track, if will appear again
          mGIDToTableID[trackIndex] = selRes;                  // negative answer means rejection
        }
      }
    }
  }
}

int FilteringSpec::processBarrelTrack(GIndex idx, const o2::globaltracking::RecoContainer& recoData)
{
  int res = selectTrack(idx, recoData);
  if (res < 0) {
    return res;
  }
  mNeedToSave = true;
  auto contributorsGID = recoData.getSingleDetectorRefs(idx);
  //
  // here is example of processing the ITS track
  if (contributorsGID[GID::ITS].isIndexSet()) {
    mITSTrackIDCache[contributorsGID[GID::ITS]] = 0;
  }

  return res;
}

bool FilteringSpec::selectTrack(GIndex id, const o2::globaltracking::RecoContainer& recoData)
{
  o2::track::TrackParCov t;
  auto src = id.getSource();
  if (src == GID::ITSTPCTRDTOF) {
    t = recoData.getTrack<o2::track::TrackParCov>(recoData.getITSTPCTRDTOFMatches()[id].getTrackRef()); // ITSTPCTRDTOF is ITSTPCTRD + TOF cluster
  } else if (src == GID::TPCTRDTOF) {
    t = recoData.getTrack<o2::track::TrackParCov>(recoData.getTPCTRDTOFMatches()[id].getTrackRef()); // TPCTRDTOF is TPCTRD + TOF cluster
  } else if (src == GID::ITSTPCTOF) {
    t = recoData.getTrack<o2::track::TrackParCov>(recoData.getTOFMatch(id).getTrackRef()); // ITSTPCTOF is ITSTPC + TOF cluster
  } else {                                                                                 // for the rest, get the track directly
    t = recoData.getTrack<o2::track::TrackParCov>(id);
  }
  // select on track kinematics, for example, on pT
  if (t.getPt() < 2.) {
    return false;
  }
  return true;
}

void FilteringSpec::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "data filtering total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void FilteringSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  static bool initOnceDone = false;
  if (!initOnceDone) { // this params need to be queried only once
    initOnceDone = true;
  }
}

DataProcessorSpec getDataFilteringSpec(GID::mask_t src, bool enableSV, bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "FILTERED_RECO_TF", 0, o2::framework::Lifetime::Sporadic); // sporadic to avoid sensing TFs which had no selected stuff

  auto dataRequest = std::make_shared<DataRequest>();

  dataRequest->requestTracks(src, useMC);
  dataRequest->requestPrimaryVertertices(useMC);
  if (src[GID::CTP]) {
    LOGF(info, "Requesting CTP digits");
    dataRequest->requestCTPDigits(useMC);
  }
  if (enableSV) {
    dataRequest->requestSecondaryVertertices(useMC);
  }
  if (src[GID::TPC]) {
    dataRequest->requestClusters(GIndex::getSourcesMask("TPC"), false); // no need to ask for TOF clusters as they are requested with TOF tracks
  }
  if (src[GID::EMC]) {
    dataRequest->requestEMCALCells(useMC);
  }

  return DataProcessorSpec{
    "reco-data-filter",
    dataRequest->inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<FilteringSpec>(src, dataRequest, enableSV, useMC)},
    Options{/*ConfigParamSpec{"reco-mctracks-only", VariantType::Int, 0, {"Store only reconstructed MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}}*/}};
}

} // namespace o2::filtering

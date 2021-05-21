// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GPUWorkflowHelper/GPUWorkflowHelper.h"
#include "DataFormatsTRD/RecoInputContainer.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "ITStracking/IOUtils.h"
#include "DataFormatsTPC/WorkflowHelper.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include <type_traits>

using namespace o2::globaltracking;
using namespace o2::gpu;

struct GPUWorkflowHelper::tmpDataContainer {
  std::vector<o2::BaseCluster<float>> ITSClustersArray;
  std::vector<int> tpcLinkITS, tpcLinkTRD, tpcLinkTOF;
  std::vector<const o2::track::TrackParCov*> globalTracks;
  std::vector<float> globalTrackTimes;
};

std::shared_ptr<const GPUWorkflowHelper::tmpDataContainer> GPUWorkflowHelper::fillIOPtr(GPUTrackingInOutPointers& ioPtr, const o2::globaltracking::RecoContainer& recoCont, bool useMC, const GPUCalibObjectsConst* calib, o2::dataformats::GlobalTrackID::mask_t maskCl, o2::dataformats::GlobalTrackID::mask_t maskTrk, o2::dataformats::GlobalTrackID::mask_t maskMatch)
{
  auto retVal = std::make_shared<tmpDataContainer>();

  if (maskCl[GID::ITS] && ioPtr.nItsClusters == 0) {
    const auto& ITSClusterROFRec = recoCont.getITSClustersROFRecords();
    const auto& clusITS = recoCont.getITSClusters();
    if (clusITS.size() && ITSClusterROFRec.size()) {
      if (calib && calib->itsPatternDict) {
        const auto& patterns = recoCont.getITSClustersPatterns();
        auto pattIt = patterns.begin();
        retVal->ITSClustersArray.reserve(clusITS.size());
        o2::its::ioutils::convertCompactClusters(clusITS, pattIt, retVal->ITSClustersArray, *calib->itsPatternDict);
        ioPtr.itsClusters = retVal->ITSClustersArray.data();
      }
      ioPtr.nItsClusters = clusITS.size();
      ioPtr.itsCompClusters = clusITS.data();
      ioPtr.nItsClusterROF = ITSClusterROFRec.size();
      ioPtr.itsClusterROF = ITSClusterROFRec.data();
      if (useMC) {
        const auto& ITSClsLabels = recoCont.mcITSClusters.get();
        ioPtr.itsClusterMC = ITSClsLabels;
      }
    }
    //LOG(info) << "Got " << ioPtr.nItsClusters << " ITS Clusters";
  }
  if (maskTrk[GID::ITS] && ioPtr.nItsTracks == 0) {
    const auto& ITSTracksArray = recoCont.getITSTracks();
    const auto& ITSTrackROFRec = recoCont.getITSTracksROFRecords();
    if (ITSTracksArray.size() && ITSTrackROFRec.size()) {
      const auto& ITSTrackClusIdx = recoCont.getITSTracksClusterRefs();
      ioPtr.nItsTracks = ITSTracksArray.size();
      ioPtr.itsTracks = ITSTracksArray.data();
      ioPtr.itsTrackClusIdx = ITSTrackClusIdx.data();
      ioPtr.nItsTrackROF = ITSTrackROFRec.size();
      ioPtr.itsTrackROF = ITSTrackROFRec.data();
      if (useMC) {
        const auto& ITSTrkLabels = recoCont.getITSTracksMCLabels();
        ioPtr.itsTrackMC = ITSTrkLabels.data();
      }
    }
    //LOG(info) << "Got " << ioPtr.nItsTracks << " ITS Tracks";
  }

  if (maskTrk[GID::ITSTPC] && ioPtr.nTracksTPCITSO2 == 0) {
    const auto& trkITSTPC = recoCont.getTPCITSTracks();
    if (trkITSTPC.size()) {
      ioPtr.nTracksTPCITSO2 = trkITSTPC.size();
      ioPtr.tracksTPCITSO2 = trkITSTPC.data();
    }
    //LOG(info) << "Got " << ioPtr.nTracksTPCITSO2 << " ITS-TPC Tracks";
  }

  if (maskCl[GID::TOF] && ioPtr.nTOFClusters == 0) {
    const auto& tofClusters = recoCont.getTOFClusters();
    if (tofClusters.size()) {
      ioPtr.nTOFClusters = tofClusters.size();
      ioPtr.tofClusters = tofClusters.data();
    }
    //LOG(info) << "Got " << ioPtr.nTOFClusters << " TOF Clusters";
  }

  if ((maskMatch[GID::TOF] || maskMatch[GID::ITSTPCTOF] || maskMatch[GID::ITSTPCTRDTOF]) && ioPtr.nTOFMatches == 0) {
    const auto& tofMatches = recoCont.getTOFMatches();
    if (tofMatches.size()) {
      ioPtr.nTOFMatches = tofMatches.size();
      ioPtr.tofMatches = tofMatches.data();
    }
    //LOG(info) << "Got " << ioPtr.nTOFMatches << " TOF Matches";
  }

  if (maskMatch[GID::TPCTOF] && ioPtr.nTPCTOFMatches == 0) {
    const auto& tpctofMatches = recoCont.getTPCTOFMatches();
    if (tpctofMatches.size()) {
      ioPtr.nTPCTOFMatches = tpctofMatches.size();
      ioPtr.tpctofMatches = tpctofMatches.data();
    }
    //LOG(info) << "Got " << ioPtr.nTPCTOFMatches << " TPC-TOF Matches";
  }

  if (maskCl[GID::TRD]) {
    recoCont.inputsTRD->fillGPUIOPtr(&ioPtr);
    //LOG(info) << "Got " << ioPtr.nTRDTracklets << " TRD Tracklets";
  }

  if (maskTrk[GID::ITSTPCTRD] && ioPtr.nTRDTracksITSTPCTRD == 0) {
    const auto& trdTracks = recoCont.getITSTPCTRDTracks<o2::trd::TrackTRD>();
    if (trdTracks.size()) {
      ioPtr.nTRDTracksITSTPCTRD = trdTracks.size();
      ioPtr.trdTracksITSTPCTRD = trdTracks.data();
    }
    //LOG(info) << "Got " << ioPtr.nTRDTracksITSTPCTRD << " ITS-TPC-TRD Tracks";
  }

  if (maskTrk[GID::TPCTRD] && ioPtr.nTRDTracksTPCTRD == 0) {
    const auto& trdTracks = recoCont.getTPCTRDTracks<o2::trd::TrackTRD>();
    if (trdTracks.size()) {
      ioPtr.nTRDTracksTPCTRD = trdTracks.size();
      ioPtr.trdTracksTPCTRD = trdTracks.data();
    }
    //LOG(info) << "Got " << ioPtr.nTRDTracksTPCTRD << " TPC-TRD Tracks";
  }

  if (maskCl[GID::TPC] && ioPtr.clustersNative == nullptr) {
    ioPtr.clustersNative = &recoCont.getTPCClusters();
    //LOG(info) << "Got " << ioPtr.clustersNative->nClustersTotal << " TPC Clusters";
  }

  if (maskTrk[GID::TPC] && ioPtr.nOutputTracksTPCO2 == 0) {
    const auto& tpcTracks = recoCont.getTPCTracks();
    const auto& tpcClusRefs = recoCont.getTPCTracksClusterRefs();
    ioPtr.outputTracksTPCO2 = tpcTracks.data();
    ioPtr.nOutputTracksTPCO2 = tpcTracks.size();
    ioPtr.outputClusRefsTPCO2 = tpcClusRefs.data();
    ioPtr.nOutputClusRefsTPCO2 = tpcClusRefs.size();
    if (useMC) {
      const auto& tpcTracksMC = recoCont.getTPCTracksMCLabels();
      ioPtr.outputTracksTPCO2MC = tpcTracksMC.data();
    }
    if (ioPtr.nTracksTPCITSO2) {
      retVal->tpcLinkITS.resize(ioPtr.nOutputTracksTPCO2, -1);
      ioPtr.tpcLinkITS = retVal->tpcLinkITS.data();
    }
    if (ioPtr.nTOFMatches || ioPtr.nTPCTOFMatches) {
      retVal->tpcLinkTOF.resize(ioPtr.nOutputTracksTPCO2, -1);
      ioPtr.tpcLinkTOF = retVal->tpcLinkTOF.data();
    }
    if (ioPtr.nTRDTracksITSTPCTRD || ioPtr.nTRDTracksTPCTRD) {
      retVal->tpcLinkTRD.resize(ioPtr.nOutputTracksTPCO2, -1);
      ioPtr.tpcLinkTRD = retVal->tpcLinkTRD.data();
    }
    //LOG(info) << "Got " << ioPtr.nOutputTracksTPCO2 << " TPC Tracks";
  }

  auto creator = [maskTrk, &ioPtr, &recoCont, &retVal](auto& trk, GID gid, float time, float) {
    if (gid.getSource() == GID::ITSTPCTOF) {
      if (maskTrk[GID::TPC] && ioPtr.nTracksTPCITSO2) {
        const auto& match = recoCont.getTOFMatch(gid);
        const auto& trkItsTPC = ioPtr.tracksTPCITSO2[match.getTrackIndex()];
        if (retVal->tpcLinkTOF.size()) {
          retVal->tpcLinkTOF[trkItsTPC.getRefTPC().getIndex()] = match.getTOFClIndex();
        }
        if (retVal->tpcLinkITS.size()) {
          retVal->tpcLinkITS[trkItsTPC.getRefTPC().getIndex()] = trkItsTPC.getRefITS().getIndex();
        }
      }
    } else if constexpr (isTPCTrack<decltype(trk)>()) {
      time = trk.getTime0();
    } else if constexpr (isTPCITSTrack<decltype(trk)>()) {
      if (maskTrk[GID::TPC] && retVal->tpcLinkITS.size()) {
        retVal->tpcLinkITS[trk.getRefTPC().getIndex()] = trk.getRefITS().getIndex();
      }
    } else if constexpr (isTPCTOFTrack<decltype(trk)>()) {
      if (maskTrk[GID::TPC] && ioPtr.nTPCTOFMatches && retVal->tpcLinkTOF.size()) {
        const auto& match = ioPtr.tpctofMatches[trk.getRefMatch()];
        retVal->tpcLinkTOF[match.getTrackIndex()] = match.getTOFClIndex();
      }
    }
    if constexpr (std::is_base_of_v<o2::track::TrackParCov, std::decay_t<decltype(trk)>>) {
      retVal->globalTracks.emplace_back(&trk);
      retVal->globalTrackTimes.emplace_back(time);
      return true;
    } else {
      return false;
    }
  };
  recoCont.createTracksVariadic(creator);
  if (maskTrk[GID::TPC] && retVal->tpcLinkTRD.size()) {
    for (unsigned int i = 0; i < ioPtr.nTRDTracksTPCTRD; i++) { // TODO: This should be handled by the createTracks logic, but so far it lacks the TRD tracks
      retVal->tpcLinkTRD[ioPtr.trdTracksTPCTRD[i].getRefGlobalTrackId().getIndex()] = i;
    }
    if (ioPtr.nTracksTPCITSO2) {
      for (unsigned int i = 0; i < ioPtr.nTRDTracksITSTPCTRD; i++) {
        retVal->tpcLinkTRD[ioPtr.tracksTPCITSO2[ioPtr.trdTracksITSTPCTRD[i].getRefGlobalTrackId().getIndex()].getRefTPC().getIndex()] = i | 0x40000000;
      }
    }
  }
  ioPtr.globalTracks = retVal->globalTracks.data();
  ioPtr.globalTrackTimes = retVal->globalTrackTimes.data();

  return std::move(retVal);
}

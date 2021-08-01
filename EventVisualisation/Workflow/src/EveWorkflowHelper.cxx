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

/// \file EveWorkflowHelper.cxx
/// \author julian.myrcha@cern.ch

#include <EveWorkflow/EveWorkflowHelper.h>
#include "DataFormatsTRD/TrackTRD.h"
#include "ITStracking/IOUtils.h"
#include "DataFormatsGlobalTracking/RecoContainerCreateTracksVariadic.h"
#include <type_traits>

using namespace o2::event_visualisation;

std::shared_ptr<const o2::event_visualisation::EveWorkflowHelper::tmpDataContainer>
  o2::event_visualisation::EveWorkflowHelper::compute(const o2::globaltracking::RecoContainer& recoCont,
                                                      const CalibObjectsConst* calib, GID::mask_t maskCl,
                                                      GID::mask_t maskTrk, GID::mask_t maskMatch)
{
  unsigned int nTracksTPCITSO2 = 0;
  unsigned int nTRDTracksITSTPCTRD = 0;
  unsigned int nTRDTracksTPCTRD = 0;
  unsigned int nITSTPCTOFMatches = 0;
  unsigned int nITSTPCTRDTOFMatches = 0;
  unsigned int nTPCTRDTOFMatches = 0;
  unsigned int nTPCTOFMatches = 0;

  auto retVal = std::make_shared<tmpDataContainer>();

  if (maskCl[GID::ITS]) {
    const auto& ITSClusterROFRec = recoCont.getITSClustersROFRecords();
    const auto& clusITS = recoCont.getITSClusters();
    if (clusITS.size() && ITSClusterROFRec.size()) {
      if (calib && calib->itsPatternDict) {
        const auto& patterns = recoCont.getITSClustersPatterns();
        auto pattIt = patterns.begin();
        retVal->ITSClustersArray.reserve(clusITS.size());
        o2::its::ioutils::convertCompactClusters(clusITS, pattIt, retVal->ITSClustersArray, *calib->itsPatternDict);
      }
    }
  }

  if (maskTrk[GID::ITSTPC] && nTracksTPCITSO2 == 0) {
    const auto& trkITSTPC = recoCont.getTPCITSTracks();
    if (trkITSTPC.size()) {
      nTracksTPCITSO2 = trkITSTPC.size();
    }
    LOG(info) << "Got " << nTracksTPCITSO2 << " ITS-TPC Tracks";
  }

  if ((maskMatch[GID::TOF] || maskMatch[GID::ITSTPCTOF])) {
    const auto& tofMatches = recoCont.getITSTPCTOFMatches();
    if (tofMatches.size()) {
      nITSTPCTOFMatches = tofMatches.size();
    }
    LOG(info) << "Got " << nITSTPCTOFMatches << " ITS-TPC-TOF Matches";
  }
  if ((maskMatch[GID::TOF] || maskMatch[GID::ITSTPCTRDTOF])) {
    const auto& tofMatches = recoCont.getITSTPCTRDTOFMatches();
    if (tofMatches.size()) {
      nITSTPCTRDTOFMatches = tofMatches.size();
    }
    LOG(info) << "Got " << nITSTPCTRDTOFMatches << " ITS-TPC-TRD-TOF Matches";
  }
  if (maskMatch[GID::TOF] || (maskMatch[GID::TPCTOF])) {
    const auto& tpctofMatches = recoCont.getTPCTOFMatches();
    if (tpctofMatches.size()) {
      nTPCTOFMatches = tpctofMatches.size();
    }
    LOG(info) << "Got " << nTPCTOFMatches << " TPC-TOF Matches";
  }
  if (maskMatch[GID::TOF] || (maskMatch[GID::TPCTRDTOF])) {
    const auto& tpctofMatches = recoCont.getTPCTRDTOFMatches();
    if (tpctofMatches.size()) {
      nTPCTRDTOFMatches = tpctofMatches.size();
    }
    LOG(info) << "Got " << nTPCTRDTOFMatches << " TPC-TRD-TOF Matches";
  }

  if (maskTrk[GID::ITSTPCTRD]) {
    const auto& trdTracks = recoCont.getITSTPCTRDTracks<o2::trd::TrackTRD>();
    if (trdTracks.size()) {
      nTRDTracksITSTPCTRD = trdTracks.size();
    }
    LOG(info) << "Got " << nTRDTracksITSTPCTRD << " ITS-TPC-TRD Tracks";
  }

  if (maskTrk[GID::TPCTRD] && nTRDTracksTPCTRD == 0) {
    const auto& trdTracks = recoCont.getTPCTRDTracks<o2::trd::TrackTRD>();
    if (trdTracks.size()) {
      nTRDTracksTPCTRD = trdTracks.size();
    }
    LOG(info) << "Got " << nTRDTracksTPCTRD << " TPC-TRD Tracks";
  }

  if (maskTrk[GID::TPC]) {
    const auto& tpcTracks = recoCont.getTPCTracks();
    const auto& tpcClusRefs = recoCont.getTPCTracksClusterRefs();
    if (nTracksTPCITSO2) {
      retVal->tpcLinkITS.resize(tpcTracks.size(), -1);
    }
    if (nITSTPCTRDTOFMatches || nITSTPCTOFMatches || nTPCTRDTOFMatches || nTPCTOFMatches) {
      retVal->tpcLinkTOF.resize(tpcTracks.size(), -1);
    }
    if (nTRDTracksITSTPCTRD || nTRDTracksTPCTRD) {
      retVal->tpcLinkTRD.resize(tpcTracks.size(), -1);
    }
    LOG(info) << "Got " << tpcTracks.size() << " TPC Tracks";
  }

  auto creator = [maskTrk, &recoCont, &retVal, nTracksTPCITSO2, nTPCTOFMatches](auto& trk, GID gid, float time, float) {
    if (gid.getSource() == GID::ITSTPCTOF) {
      if (maskTrk[GID::TPC] && nTracksTPCITSO2) {
        const auto& match = recoCont.getTOFMatch(gid);
        const auto& trkITSTPC = recoCont.getTPCITSTracks();
        const auto tracksTPCITSO2 = trkITSTPC.data();
        const auto& trkItsTPC = tracksTPCITSO2[match.getTrackIndex()];
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
      if (maskTrk[GID::TPC] && nTPCTOFMatches && retVal->tpcLinkTOF.size()) {
        const auto& tpctofMatches = recoCont.getTPCTOFMatches();
        const auto tpctofMatchesData = tpctofMatches.data();
        const auto& match = tpctofMatchesData[trk.getRefMatch()];
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
    const auto& trdTracks = recoCont.getTPCTRDTracks<o2::trd::TrackTRD>();
    const auto trdTracksTPCTRD = trdTracks.data();
    const auto nTRDTracksTPCTRD = trdTracks.size();
    for (unsigned int i = 0; i < nTRDTracksTPCTRD; i++) { // TODO: This should be handled by the createTracks logic, but so far it lacks the TRD tracks
      retVal->tpcLinkTRD[trdTracksTPCTRD[i].getRefGlobalTrackId().getIndex()] = i;
    }
    if (nTracksTPCITSO2) {
      const auto& trdTracks = recoCont.getITSTPCTRDTracks<o2::trd::TrackTRD>();
      const auto trdTracksITSTPCTRD = trdTracks.data();

      const auto& trkITSTPC = recoCont.getTPCITSTracks();
      const auto tracksTPCITSO2 = trkITSTPC.data();

      for (unsigned int i = 0; i < nTRDTracksITSTPCTRD; i++) {
        retVal->tpcLinkTRD[tracksTPCITSO2[trdTracksITSTPCTRD[i].getRefGlobalTrackId().getIndex()].getRefTPC().getIndex()] = i | 0x40000000;
      }
    }
  }

  return std::move(retVal);
}

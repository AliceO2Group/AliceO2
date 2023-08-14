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

/// \file TrackCuts.h
/// \brief Class to perform track cuts

// This class is developed in order to be used for extensive track selections. The selections are done detector wise
//(or for detector combinations).
// sources: https://github.com/AliceO2Group/AliceO2/blob/e988b0c43346ccb24f3515d1a24f058313f14a0f/DataFormats/Reconstruction/include/ReconstructionDataFormats/GlobalTrackID.h#L40
//
// !!! For further development:
// The main method is isSelected(o2::dataformats::GlobalTrackID, o2::globaltracking::RecoContainer&), which is a boolean
// that returns true only if all the checks are passed. First, based on the global track id, the track detector source is
// inquired(e.g. below for ITS and TPC tracks). The source-specific tracks is initialized in order to access the
// source-specific parameters than one wants to perform selections on.
// For each detector source, the inquiry should be done in such way that “false” should be returned if the checks are not passed,
// moving to the next detector source otherwise.
// (e.g below for TPC tracks, where the track selections used here: https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/GlobalTracking/src/MatchITSTPCQC.cxx#L318
// are reproduced;
// Moreover, an example of how this class should be used can be found here: https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/GlobalTracking/src/MatchITSTPCQC.cxx#L184).

#ifndef O2_TRACK_CUTS_STUDY_H
#define O2_TRACK_CUTS_STUDY_H

#include "Framework/Logger.h"
#include "Framework/DataTypes.h"
#include "ReconstructionDataFormats/Track.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTRD/TrackTRD.h"
#include "ITSStudies/TrackMethods.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include <set>
#include <vector>
#include "Rtypes.h"
#include <gsl/span>
namespace o2
{
namespace its
{
namespace study
{
using GID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
class TrackCuts
{
 public:
  void countTPCClusters(const o2::globaltracking::RecoContainer& data)
  {
    const auto& tpcTracks = data.getTPCTracks();
    const auto& tpcClusRefs = data.getTPCTracksClusterRefs();
    const auto& tpcClusShMap = data.clusterShMapTPC;
    const auto& tpcClusAcc = data.getTPCClusters();
    constexpr int maxRows = 152;
    constexpr int neighbour = 2;
    int ntr = tpcTracks.size();
    mTPCCounters.clear();
    mTPCCounters.resize(ntr);
    for (int itr = 0; itr < ntr; itr++) {
      std::array<bool, maxRows> clMap{}, shMap{};
      uint8_t sectorIndex, rowIndex;
      uint32_t clusterIndex;
      auto& counters = mTPCCounters[itr];
      const auto& track = tpcTracks[itr];
      for (int i = 0; i < track.getNClusterReferences(); i++) {
        o2::tpc::TrackTPC::getClusterReference(tpcClusRefs, i, sectorIndex, rowIndex, clusterIndex, track.getClusterRef());
        unsigned int absoluteIndex = tpcClusAcc.clusterOffset[sectorIndex][rowIndex] + clusterIndex;
        clMap[rowIndex] = true;
        if (tpcClusShMap[absoluteIndex] & GPUCA_NAMESPACE::gpu::GPUTPCGMMergedTrackHit::flagShared) {
          if (!shMap[rowIndex]) {
            counters.shared++;
          }
          shMap[rowIndex] = true;
        }
      }
      int last = -1;
      for (int i = 0; i < maxRows; i++) {
        if (clMap[i]) {
          counters.crossed++;
          counters.found++;
          last = i;
        } else if ((i - last) <= neighbour) {
          counters.crossed++;
        } else {
          int lim = std::min(i + 1 + neighbour, maxRows);
          for (int j = i + 1; j < lim; j++) {
            if (clMap[j]) {
              counters.crossed++;
              break;
            }
          }
        }
      }
    }
  }
  //////////////////////////////// O2 ////////////////////////////////
  bool isSelected(GID trackIndex, o2::globaltracking::RecoContainer& data)
  {
    o2::track::TrackParCov trk;
    auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
    auto src = trackIndex.getSource(); // make selections depending on track source
    // ITS tracks
    if (contributorsGID[GIndex::Source::ITS].isIndexSet()) { // ITS tracks selection
      isBarrelTrack = true;
      const auto& itsTrk = data.getITSTrack(contributorsGID[GID::ITS]);
      int ITSnClusters = itsTrk.getNClusters();
      float ITSchi2 = itsTrk.getChi2();
      float itsChi2NCl = ITSnClusters != 0 ? ITSchi2 / (float)ITSnClusters : 0;
      uint8_t itsClusterMap = itsTrk.getPattern();
      SetRequireHitsInITSLayers(1, {0, 1, 2});
      if (itsChi2NCl >= mMaxChi2PerClusterITS ||
          TrackMethods::FulfillsITSHitRequirements(itsClusterMap, mRequiredITSHits) == false) {
        // LOGP(info,"FAILURE hits in ITS layers");
        return false;
      }
    }
    // TPC tracks
    /* if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
      LOGP(info, "****** INSIDE TPC ********");
      //countTPCClusters(data);
      isBarrelTrack = true;
      const auto& tpcTrk = data.getTPCTrack(contributorsGID[GID::TPC]);
      const auto& tpcClData = mTPCCounters[contributorsGID[GID::TPC]];
      math_utils::Point3D<float> v{};
      std::array<float, 2> dca;
      int tpcNClsFindable = tpcTrk.getNClusters();
      float tpcChi2NCl = tpcTrk.getNClusters() ? tpcTrk.getChi2() / tpcTrk.getNClusters() : 0;
      double tpcNClsFindableMinusFound = tpcTrk.getNClusters() - tpcClData.found;
      double tpcNClsFindableMinusCrossedRows = tpcTrk.getNClusters() - tpcClData.crossed;
      float tpcNClsCrossedRows = (float)((int16_t)tpcNClsFindable - tpcNClsFindableMinusCrossedRows)/tpcNClsFindable;
      float tpcCrossedRowsOverFindableCl = (float)tpcNClsCrossedRows / (float)tpcNClsFindable;
      if(tpcCrossedRowsOverFindableCl < 0.8) {
        LOGP(info,"FAILURE tpcCrossedRowsOverFindableCl");
        return false;
      }
      if(tpcClData.crossed < 70){
        LOGP(info,"FAILURE crossed");
        return false;
      }
      if(tpcChi2NCl >= mMaxChi2PerClusterTPC){
        LOGP(info,"FAILURE tpcChi2NCl");
        return false;
      }
    } */
    if (isBarrelTrack) { // track selection for barrel tracks
      trk = data.getTrackParam(trackIndex);
      if (trk.getPt() < mMinPt && trk.getPt() > mMaxPt && trk.getEta() < mMinEta && trk.getEta() > mMaxEta) {
        return false;
      }
    }
    return true;
  }

  void SetRequireHitsInITSLayers(int8_t minNRequiredHits, std::set<uint8_t> requiredLayers)
  {
    // layer 0 corresponds to the the innermost ITS layer
    mRequiredITSHits.push_back(std::make_pair(minNRequiredHits, requiredLayers));
    // LOG(info) << "Track selection, set require hits in ITS layers: " << static_cast<int>(minNRequiredHits);
  }

 private:
  // counters for TPC clusters
  struct TPCCounters {
    uint8_t shared = 0;
    uint8_t found = 0;
    uint8_t crossed = 0;
  };
  std::vector<TPCCounters> mTPCCounters;

  bool isBarrelTrack = true; // all barrel tracks must have either ITS or TPC contribution -> true if ITS || TPC track source condition is passed
  // cut values
  float mPtTPCCut = 0.1f;
  float mEtaTPCCut = 1.4f;
  int32_t mNTPCClustersCut = 60;
  float mDCACut = 100.f;
  float mDCACutY = 10.f;
  // kinematic cuts
  float mMinPt{0.1f}, mMaxPt{1e10f}; // range in pT
  float mMinEta{0.8}, mMaxEta{0.8};  // range in eta
  float mBz = o2::base::Propagator::Instance()->getNominalBz();

  float mMaxChi2PerClusterITS{36.0}; // max its fit chi2 per ITS cluster
  float mMaxChi2PerClusterTPC{4.0};  // max its fit chi2 per ITS cluster
  int mMinNClustersITS{0};           // min number of ITS clusters

  // vector of ITS requirements (minNRequiredHits in specific requiredLayers)
  int8_t minNRequiredHits = 1;
  std::set<uint8_t> requiredLayers{0, 1, 2}; // one hit in the first three layers
  std::vector<std::pair<int8_t, std::set<uint8_t>>> mRequiredITSHits{};

  ClassDefNV(TrackCuts, 1);
};
} // namespace study
} // namespace its
} // namespace o2

#endif

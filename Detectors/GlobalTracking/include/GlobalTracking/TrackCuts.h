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
/// \author amelia.lindner@cern.ch

#ifndef ALICEO2_TRACKCUTS_H
#define ALICEO2_TRACKCUTS_H

#include "Framework/Logger.h"
#include "Framework/DataTypes.h"
#include "ReconstructionDataFormats/Track.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "GlobalTracking/TrackMethods.h"
#include <set>
#include <vector>
#include "Rtypes.h"
namespace o2
{
using GID = o2::dataformats::GlobalTrackID;
class TrackCuts
{
 public:
  //////////////////////////////// O2 ////////////////////////////////
  bool isSelected(GID trackIndex, o2::globaltracking::RecoContainer& data)
  {
    o2::track::TrackParCov trk;
    auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
    auto src = trackIndex.getSource(); //make selections depending on track source
    if (src == GID::ITSTPCTRDTOF) {
      trk = data.getTrack<o2::track::TrackParCov>(data.getITSTPCTRDTOFMatches()[trackIndex].getTrackRef()); // ITSTPCTRDTOF is ITSTPCTRD + TOF cluster
    } else if (src == GID::TPCTRDTOF) {
      trk = data.getTrack<o2::track::TrackParCov>(data.getTPCTRDTOFMatches()[trackIndex].getTrackRef()); // TPCTRDTOF is TPCTRD + TOF cluster
    } else if (src == GID::ITSTPCTOF) {
      trk = data.getTrack<o2::track::TrackParCov>(data.getTOFMatch(trackIndex).getTrackRef()); // ITSTPCTOF is ITSTPC + TOF cluster
    } else if (src == GID::TPC) { //TPC tracks selection
      const auto& tpcTrk = data.getTPCTrack(src);
      uint8_t tpcNClsShared, tpcNClsFound, tpcNClsCrossed, tpcNClsFindable, tpcChi2NCl;
      tpcNClsFindable = tpcTrk.getNClusters();
      tpcChi2NCl = tpcTrk.getNClusters() ? tpcTrk.getChi2() / tpcTrk.getNClusters() : 0;
      TrackMethods::countTPCClusters(tpcTrk, data.getTPCTracksClusterRefs(), data.clusterShMapTPC, data.getTPCClusters(), tpcNClsShared, tpcNClsFound, tpcNClsCrossed);
      double tpcCrossedRowsOverFindableCls = tpcNClsCrossed / tpcNClsFindable;
      if (tpcTrk.getPt() >= mMinPt && tpcTrk.getPt() <= mMaxPt && 
          tpcTrk.getEta() >= mMinEta && tpcTrk.getEta() <= mMaxEta &&
          tpcNClsFound >= mMinNClustersTPC &&
          tpcNClsCrossed >= mMinNCrossedRowsTPC &&
          tpcCrossedRowsOverFindableCls >= mMinNCrossedRowsOverFindableClustersTPC &&
          tpcChi2NCl <= mMaxChi2PerClusterTPC) {
        return true;
      } else {
        return false;
      }
    } else if (src == GID::ITS) { //ITS tracks selection
      const auto& itsTrk = data.getITSTrack(contributorsGID[src]);
      int ITSnClusters = itsTrk.getNClusters();
      float ITSchi2 = itsTrk.getChi2();
      float itsChi2NCl = ITSnClusters != 0 ? ITSchi2 / (float)ITSnClusters : 0;
      uint8_t itsClusterMap = itsTrk.getPattern();
      if (itsTrk.getPt() >= mMinPt && itsTrk.getPt() <= mMaxPt &&
          itsTrk.getEta() >= mMinEta && itsTrk.getEta() <= mMaxEta &&
          ITSnClusters >= mMinNClustersITS &&
          itsChi2NCl <= mMaxChi2PerClusterITS &&
          TrackMethods::FulfillsITSHitRequirements(itsClusterMap, mRequiredITSHits)) {
        return true;
      } else {
        return false;
      }
    } else { // for the rest, get the track directly
      trk = data.getTrack<o2::track::TrackParCov>(trackIndex);
    }
    if (trk.getPt() >= mMinPt && trk.getPt() <= mMaxPt &&
        trk.getEta() >= mMinEta && trk.getEta() <= mMaxEta) {
      return true;
    } else {
      return false;
    }
  }

  //////////////////////////////// O2Physics ////////////////////////////////

  enum class TrackCutsList : int {
    kTrackType = 0,
    kPtRange,
    kEtaRange,
    kTPCNCls,
    kTPCCrossedRows,
    kTPCCrossedRowsOverNCls,
    kTPCChi2NDF,
    kTPCRefit,
    kITSNCls,
    kITSChi2NDF,
    kITSRefit,
    kITSHits,
    kGoldenChi2,
    kDCAxy,
    kDCAz,
    kNCuts
  };

  static const std::string mCutNames[static_cast<int>(TrackCutsList::kNCuts)];
  // Temporary function to check if track passes selection criteria. To be replaced by framework filters.
  template <typename T>
  bool IsSelected(T const& track)
  {
    const bool isRun2 = track.trackType() == o2::aod::track::Run2Track || track.trackType() == o2::aod::track::Run2Tracklet;
    if (track.trackType() == mTrackType &&
        track.pt() >= mMinPt && track.pt() <= mMaxPt &&
        track.eta() >= mMinEta && track.eta() <= mMaxEta &&
        track.tpcNClsFound() >= mMinNClustersTPC &&
        track.tpcNClsCrossedRows() >= mMinNCrossedRowsTPC &&
        track.tpcCrossedRowsOverFindableCls() >= mMinNCrossedRowsOverFindableClustersTPC &&
        (track.itsNCls() >= mMinNClustersITS) &&
        (track.itsChi2NCl() <= mMaxChi2PerClusterITS) &&
        (track.tpcChi2NCl() <= mMaxChi2PerClusterTPC) &&
        (mRequireITSRefit ? (isRun2 ? (track.flags() & o2::aod::track::ITSrefit) : track.hasITS()) : true) &&
        (mRequireTPCRefit ? (isRun2 ? (track.flags() & o2::aod::track::TPCrefit) : track.hasTPC()) : true) &&
        ((isRun2 && mRequireGoldenChi2) ? (track.flags() & o2::aod::track::GoldenChi2) : true) &&
        TrackMethods::FulfillsITSHitRequirements(track.itsClusterMap(), mRequiredITSHits) &&
        abs(track.dcaXY()) <= ((mMaxDcaXYPtDep) ? mMaxDcaXYPtDep(track.pt()) : mMaxDcaXY) &&
        abs(track.dcaZ()) <= mMaxDcaZ) {
      return true;
    } else {
      return false;
    }
  }

  // Temporary function to check if track passes a given selection criteria. To be replaced by framework filters.
  template <typename T>
  bool IsSelected(T const& track, const TrackCutsList& cut)
  {
    const bool isRun2 = track.trackType() == o2::aod::track::Run2Track || track.trackType() == o2::aod::track::Run2Tracklet;

    switch (cut) {
      case TrackCutsList::kTrackType:
        return track.trackType() == mTrackType;

      case TrackCutsList::kPtRange:
        return track.pt() >= mMinPt && track.pt() <= mMaxPt;

      case TrackCutsList::kEtaRange:
        return track.eta() >= mMinEta && track.eta() <= mMaxEta;

      case TrackCutsList::kTPCNCls:
        return track.tpcNClsFound() >= mMinNClustersTPC;

      case TrackCutsList::kTPCCrossedRows:
        return track.tpcNClsCrossedRows() >= mMinNCrossedRowsTPC;

      case TrackCutsList::kTPCCrossedRowsOverNCls:
        return track.tpcCrossedRowsOverFindableCls() >= mMinNCrossedRowsOverFindableClustersTPC;

      case TrackCutsList::kTPCChi2NDF:
        return track.tpcChi2NCl() <= mMaxChi2PerClusterTPC;

      case TrackCutsList::kTPCRefit:
        return (isRun2 && mRequireTPCRefit) ? (track.flags() & o2::aod::track::TPCrefit) : true;

      case TrackCutsList::kITSNCls:
        return track.itsNCls() >= mMinNClustersITS;

      case TrackCutsList::kITSChi2NDF:
        return track.itsChi2NCl() <= mMaxChi2PerClusterITS;

      case TrackCutsList::kITSRefit:
        return (isRun2 && mRequireITSRefit) ? (track.flags() & o2::aod::track::ITSrefit) : true;

      case TrackCutsList::kITSHits:
        return FulfillsITSHitRequirements(track.itsClusterMap());

      case TrackCutsList::kGoldenChi2:
        return (isRun2 && mRequireGoldenChi2) ? (track.flags() & o2::aod::track::GoldenChi2) : true;

      case TrackCutsList::kDCAxy:
        return abs(track.dcaXY()) <= ((mMaxDcaXYPtDep) ? mMaxDcaXYPtDep(track.pt()) : mMaxDcaXY);

      case TrackCutsList::kDCAz:
        return abs(track.dcaZ()) <= mMaxDcaZ;

      default:
        return false;
    }
  }

  void SetTrackType(o2::aod::track::TrackTypeEnum trackType) { mTrackType = trackType; }
  void SetPtRange(float minPt = 0.f, float maxPt = 1e10f)
  {
    mMinPt = minPt;
    mMaxPt = maxPt;
  }
  void SetEtaRange(float minEta = -1e10f, float maxEta = 1e10f)
  {
    mMinEta = minEta;
    mMaxEta = maxEta;
  }
  void SetRequireITSRefit(bool requireITSRefit = true)
  {
    mRequireITSRefit = requireITSRefit;
  }
  void SetRequireTPCRefit(bool requireTPCRefit = true)
  {
    mRequireTPCRefit = requireTPCRefit;
  }
  void SetRequireGoldenChi2(bool requireGoldenChi2 = true)
  {
    mRequireGoldenChi2 = requireGoldenChi2;
  }
  void SetMinNClustersTPC(int minNClustersTPC)
  {
    mMinNClustersTPC = minNClustersTPC;
  }
  void SetMinNCrossedRowsTPC(int minNCrossedRowsTPC)
  {
    mMinNCrossedRowsTPC = minNCrossedRowsTPC;
  }
  void SetMinNCrossedRowsOverFindableClustersTPC(float minNCrossedRowsOverFindableClustersTPC)
  {
    mMinNCrossedRowsOverFindableClustersTPC = minNCrossedRowsOverFindableClustersTPC;
  }
  void SetMinNClustersITS(int minNClustersITS)
  {
    mMinNClustersITS = minNClustersITS;
  }
  void SetMaxChi2PerClusterTPC(float maxChi2PerClusterTPC)
  {
    mMaxChi2PerClusterTPC = maxChi2PerClusterTPC;
  }
  void SetMaxChi2PerClusterITS(float maxChi2PerClusterITS)
  {
    mMaxChi2PerClusterITS = maxChi2PerClusterITS;
  }
  void SetMaxDcaXY(float maxDcaXY) { mMaxDcaXY = maxDcaXY; }
  void SetMaxDcaZ(float maxDcaZ) { mMaxDcaZ = maxDcaZ; }

  void SetMaxDcaXYPtDep(std::function<float(float)> ptDepCut)
  {
    mMaxDcaXYPtDep = ptDepCut;
  }
  void SetRequireHitsInITSLayers(int8_t minNRequiredHits, std::set<uint8_t> requiredLayers)
  {
    // layer 0 corresponds to the the innermost ITS layer
    mRequiredITSHits.push_back(std::make_pair(minNRequiredHits, requiredLayers));
  }
  void SetRequireNoHitsInITSLayers(std::set<uint8_t> excludedLayers)
  {
    mRequiredITSHits.push_back(std::make_pair(-1, excludedLayers));
  }
  void ResetITSRequirements() { mRequiredITSHits.clear(); }

 private:
  o2::aod::track::TrackTypeEnum mTrackType{o2::aod::track::TrackTypeEnum::Track};

  // kinematic cuts
  float mMinPt{5},
    mMaxPt{6};                       // range in pT
  float mMinEta{-1e10f}, mMaxEta{1e10f}; // range in eta

  // track quality cuts
  int mMinNClustersTPC{0};                            // min number of TPC clusters
  int mMinNCrossedRowsTPC{0};                         // min number of crossed rows in TPC
  int mMinNClustersITS{0};                            // min number of ITS clusters
  float mMaxChi2PerClusterTPC{1e10f};                 // max tpc fit chi2 per TPC cluster
  float mMaxChi2PerClusterITS{1e10f};                 // max its fit chi2 per ITS cluster
  float mMinNCrossedRowsOverFindableClustersTPC{0.f}; // min ratio crossed rows / findable clusters

  float mMaxDcaXY{1e10f};                       // max dca in xy plane
  float mMaxDcaZ{1e10f};                        // max dca in z direction
  std::function<float(float)> mMaxDcaXYPtDep{}; // max dca in xy plane as function of pT

  bool mRequireITSRefit{false};   // require refit in ITS
  bool mRequireTPCRefit{false};   // require refit in TPC
  bool mRequireGoldenChi2{false}; // require golden chi2 cut (Run 2 only)

  // vector of ITS requirements (minNRequiredHits in specific requiredLayers)
  std::vector<std::pair<int8_t, std::set<uint8_t>>> mRequiredITSHits{};

  ClassDefNV(TrackCuts, 1);
};
} // namespace o2

#endif

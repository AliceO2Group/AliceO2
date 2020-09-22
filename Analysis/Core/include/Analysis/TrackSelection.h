// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
// Class for track selection
//

#ifndef TrackSelection_H
#define TrackSelection_H

#include "Framework/Logger.h"
#include "Framework/DataTypes.h"
#include <set>
#include "TObject.h"

class TrackSelection : public TObject
{
 public:
  TrackSelection() = default;

  // Temporary function to check if track passes selection criteria. To be
  // replaced by framework filters
  template <typename T>
  bool IsSelected(T const& track)
  {
    if (track.trackType() == mTrackType &&
        track.pt() >= mMinPt && track.pt() <= mMaxPt && track.eta() >= mMinEta &&
        track.eta() <= mMaxEta && track.tpcNClsFound() >= mMinNClustersTPC &&
        track.tpcNClsCrossedRows() >= mMinNCrossedRowsTPC &&
        track.tpcCrossedRowsOverFindableCls() >=
          mMinNCrossedRowsOverFindableClustersTPC &&
        (track.itsNCls() >= mMinNClustersITS) &&
        (track.itsChi2NCl() <= mMaxChi2PerClusterITS) &&
        (track.tpcChi2NCl() <= mMaxChi2PerClusterTPC) &&
        ((mRequireITSRefit) ? (track.flags() & 0x4) : true) &&
        ((mRequireTPCRefit) ? (track.flags() & 0x40) : true) &&
        ((mRequireTOF) ? ((track.flags() & 0x2000) && (track.flags() & 0x80000000)) : true) &&
        FulfillsITSHitRequirements(track.itsClusterMap()) &&
        abs(track.dcaXY()) <= ((mMaxDcaXYPtDep) ? mMaxDcaXYPtDep(track.pt()) : mMaxDcaXY) &&
        abs(track.dcaZ()) < mMaxDcaZ) {
      return true;
    } else {
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
  void SetRequireTOF(bool requireTOF = true) { mRequireTOF = requireTOF; }
  void SetMinNClustersTPC(int minNClustersTPC)
  {
    mMinNClustersTPC = minNClustersTPC;
  }
  void SetMinNCrossedRowsTPC(int minNCrossedRowsTPC)
  {
    mMinNCrossedRowsTPC = minNCrossedRowsTPC;
  }
  void SetMinNCrossedRowsOverFindableClustersTPC(
    float minNCrossedRowsOverFindableClustersTPC)
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
    if (minNRequiredHits > requiredLayers.size()) {
      LOGF(FATAL, "More ITS hits required than layers specified.");
    } else {
      mRequiredITSHits.push_back(std::make_pair(minNRequiredHits, requiredLayers));
    }
  };
  void SetRequireNoHitsInITSLayers(std::set<uint8_t> excludedLayers)
  {
    mRequiredITSHits.push_back(std::make_pair(-1, excludedLayers));
  };
  void ResetITSRequirements() { mRequiredITSHits.clear(); };

 private:
  bool FulfillsITSHitRequirements(uint8_t itsClusterMap);

  o2::aod::track::TrackTypeEnum mTrackType{o2::aod::track::TrackTypeEnum::GlobalTrack};

  // kinematic cuts
  float mMinPt{0.f}, mMaxPt{1e10f};      // range in pT
  float mMinEta{-1e10f}, mMaxEta{1e10f}; // range in eta

  // track quality cuts
  int mMinNClustersTPC{0};                          // min number of TPC clusters
  int mMinNCrossedRowsTPC{0};                       // min number of crossed rows in TPC
  int mMinNClustersITS{0};                          // min number of ITS clusters
  float mMaxChi2PerClusterTPC{1e10f};               // max tpc fit chi2 per TPC cluster
  float mMaxChi2PerClusterITS{1e10f};               // max its fit chi2 per ITS cluster
  float mMinNCrossedRowsOverFindableClustersTPC{0}; // min ratio crossed rows / findable clusters

  float mMaxDcaXY{1e10f};
  float mMaxDcaZ{1e10f};
  std::function<float(float)> mMaxDcaXYPtDep{};

  bool mRequireITSRefit{false}; // require refit in ITS
  bool mRequireTPCRefit{false}; // require refit in TPC
  bool mRequireTOF{false};      // require that track exits the TOF and that it has an associated time measurement (kTIME and kTOFOUT)

  // vector of ITS requirements (minNRequiredHits in specific requiredLayers)
  std::vector<std::pair<int8_t, std::set<uint8_t>>> mRequiredITSHits{};

  ClassDef(TrackSelection, 1);
};

#endif

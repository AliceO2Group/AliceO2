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

#include "Analysis/TrackSelectionTables.h"
#include "Framework/AnalysisTask.h"
#include "TObject.h"

class TrackSelection : public TObject {

public:
  TrackSelection();

  // Temporary function to check if track passes selection criteria. To be
  // replaced by framework filters
  bool IsSelected(o2::soa::Join<o2::aod::Tracks, o2::aod::TracksCov,
                                o2::aod::TracksExtra>::iterator const &track);

  void SetMinPt(float minPt) { mMinPt = minPt; }
  void SetMaxPt(float maxPt) { mMaxPt = maxPt; }
  void SetMinEta(float minEta) { mMinEta = minEta; }
  void SetMaxEta(float maxEta) { mMaxEta = maxEta; }
  void SetRequireITSRefit(bool requireITSRefit = true) {
    mRequireITSRefit = requireITSRefit;
  }
  void SetRequireTPCRefit(bool requireTPCRefit = true) {
    mRequireTPCRefit = requireTPCRefit;
  }
  void SetMinNClustersTPC(int minNClustersTPC) {
    mMinNClustersTPC = minNClustersTPC;
  }
  void SetMinNCrossedRowsTPC(int minNCrossedRowsTPC) {
    mMinNCrossedRowsTPC = minNCrossedRowsTPC;
  }
  void SetMinNCrossedRowsOverFindableClustersTPC(
      float minNCrossedRowsOverFindableClustersTPC) {
    mMinNCrossedRowsOverFindableClustersTPC =
        minNCrossedRowsOverFindableClustersTPC;
  }
  void SetMinNClustersITS(int minNClustersITS) {
    mMinNClustersITS = minNClustersITS;
  }
  void SetMaxChi2PerClusterTPC(float maxChi2PerClusterTPC) {
    mMaxChi2PerClusterTPC = maxChi2PerClusterTPC;
  }
  void SetMaxChi2PerClusterITS(float maxChi2PerClusterITS) {
    mMaxChi2PerClusterITS = maxChi2PerClusterITS;
  }
  void SetMaxDcaXY(float maxDcaXY) { mMaxDcaXY = maxDcaXY; }
  void SetMaxDcaZ(float maxDcaZ) { mMaxDcaZ = maxDcaZ; }
  void SetRequireHitsInITSLayers(int8_t minNRequiredHits,
                                 std::set<uint8_t> requiredLayers) {
    // layer 0 corresponds to the the innermost ITS layer
    if (minNRequiredHits > requiredLayers.size()) {
      LOG(FATAL) << "More ITS hits required than layers specified.";
    } else {
      mRequiredITSHits.push_back(
          std::make_pair(minNRequiredHits, requiredLayers));
    }
  };
  void SetRequireNoHitsInITSLayers(std::set<uint8_t> excludedLayers) {
    mRequiredITSHits.push_back(std::make_pair(-1, excludedLayers));
  };
  void ResetITSRequirements() { mRequiredITSHits.clear(); };

private:
  bool FulfillsITSHitRequirements(uint8_t itsClusterMap);

  // kinematic cuts
  float mMinPt, mMaxPt;   // range in pT
  float mMinEta, mMaxEta; // range in eta

  // track quality cuts
  int mMinNClustersTPC;        // min number of TPC clusters
  int mMinNCrossedRowsTPC;     // min number of crossed rows in TPC
  int mMinNClustersITS;        // min number of ITS clusters
  float mMaxChi2PerClusterTPC; // max tpc fit chi2 per TPC cluster
  float mMaxChi2PerClusterITS; // max its fit chi2 per ITS cluster
  float mMinNCrossedRowsOverFindableClustersTPC; // min ratio crossed rows /
                                                 // findable clusters

  float mMaxDcaXY;
  float mMaxDcaZ;

  bool mRequireITSRefit; // require refit in ITS
  bool mRequireTPCRefit; // require refit in TPC

  std::vector<std::pair<int8_t, std::set<uint8_t>>>
      mRequiredITSHits; // vector of ITS requirements (minNRequiredHits in
                        // specific requiredLayers)
};

#endif

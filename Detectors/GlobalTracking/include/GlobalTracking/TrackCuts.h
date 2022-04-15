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
#include "DetectorsBase/Propagator.h"
#include <set>
#include <vector>
#include "Rtypes.h"
#include <gsl/span>
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
    auto src = trackIndex.getSource(); // make selections depending on track source
    if (src == GID::ITSTPCTRDTOF) {
      trk = data.getTrack<o2::track::TrackParCov>(data.getITSTPCTRDTOFMatches()[trackIndex].getTrackRef()); // ITSTPCTRDTOF is ITSTPCTRD + TOF cluster
    } else if (src == GID::TPCTRDTOF) {
      trk = data.getTrack<o2::track::TrackParCov>(data.getTPCTRDTOFMatches()[trackIndex].getTrackRef()); // TPCTRDTOF is TPCTRD + TOF cluster
    } else if (src == GID::ITSTPCTOF) {
      trk = data.getTrack<o2::track::TrackParCov>(data.getTOFMatch(trackIndex).getTrackRef()); // ITSTPCTOF is ITSTPC + TOF cluster
    } else if (src == GID::TPC) {                                                              // TPC tracks selection
      const auto& tpcTrk = data.getTPCTrack(contributorsGID[GID::TPC]);
      uint8_t tpcNClsShared, tpcNClsFound, tpcNClsCrossed, tpcNClsFindable, tpcChi2NCl;
      // tpcNClsFindable = tpcTrk.getNClusters();
      // tpcChi2NCl = tpcTrk.getNClusters() ? tpcTrk.getChi2() / tpcTrk.getNClusters() : 0;
      // o2::TrackMethods::countTPCClusters(tpcTrk, data.getTPCTracksClusterRefs(), data.clusterShMapTPC, data.getTPCClusters(), tpcNClsShared, tpcNClsFound, tpcNClsCrossed);
      // double tpcCrossedRowsOverFindableCls = tpcNClsCrossed / tpcNClsFindable;
      math_utils::Point3D<float> v{};
      std::array<float, 2> dca;
      if (tpcTrk.getPt() < mPtTPCCut ||
            std::abs(tpcTrk.getEta()) > mEtaTPCCut ||
            tpcTrk.getNClusters() < mNTPCClustersCut ||
            (!(const_cast<o2::tpc::TrackTPC&>(tpcTrk).propagateParamToDCA(v, mBz, &dca, mDCACut)) ||
          std::abs(dca[0]) > mDCACutY)) {
        return false;
      } else {
        return true;
      }
    } else if (src == GID::ITS) { // ITS tracks selection
      const auto& itsTrk = data.getITSTrack(contributorsGID[GID::ITS]);
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

 private:
  // cut values
  float mPtTPCCut = 0.1f;
  float mEtaTPCCut = 1.4f;
  int32_t mNTPCClustersCut = 60;
  float mDCACut = 100.f;
  float mDCACutY = 10.f;
  // kinematic cuts
  float mMinPt{0.f},
    mMaxPt{1e10f};                       // range in pT
  float mMinEta{-1e10f}, mMaxEta{1e10f}; // range in eta
  float mBz = o2::base::Propagator::Instance()->getNominalBz();

  float mMaxChi2PerClusterITS{1e10f}; // max its fit chi2 per ITS cluster
  int mMinNClustersITS{0};            // min number of ITS clusters

  // vector of ITS requirements (minNRequiredHits in specific requiredLayers)
  std::vector<std::pair<int8_t, std::set<uint8_t>>> mRequiredITSHits{};

  ClassDefNV(TrackCuts, 1);
};
} // namespace o2

#endif

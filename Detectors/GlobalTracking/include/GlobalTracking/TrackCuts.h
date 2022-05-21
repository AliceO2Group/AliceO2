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
#include "DataFormatsTRD/TrackTRD.h"
#include "GlobalTracking/TrackMethods.h"
#include "DetectorsBase/Propagator.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include <set>
#include <vector>
#include "Rtypes.h"
#include <gsl/span>
namespace o2
{
using GID = o2::dataformats::GlobalTrackID;
using GIndex = o2::dataformats::VtxTrackIndex;
class TrackCuts
{
 public:
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
      if (ITSnClusters <= mMinNClustersITS ||
          itsChi2NCl >= mMaxChi2PerClusterITS ||
          TrackMethods::FulfillsITSHitRequirements(itsClusterMap, mRequiredITSHits) == false) {
        return false;
      }
    }
    // TPC tracks
    if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
      isBarrelTrack = true;
      const auto& tpcTrk = data.getTPCTrack(contributorsGID[GID::TPC]);
      math_utils::Point3D<float> v{};
      std::array<float, 2> dca;
      if (tpcTrk.getPt() < mPtTPCCut ||
          std::abs(tpcTrk.getEta()) > mEtaTPCCut ||
          tpcTrk.getNClusters() < mNTPCClustersCut ||
          (!(const_cast<o2::tpc::TrackTPC&>(tpcTrk).propagateParamToDCA(v, mBz, &dca, mDCACut)) ||
           std::abs(dca[0]) > mDCACutY)) {
        return false;
      }
    }
    if (isBarrelTrack) { // track selection for barrel tracks
      trk = data.getTrackParam(trackIndex);
      if (trk.getPt() < mMinPt && trk.getPt() > mMaxPt && trk.getEta() < mMinEta && trk.getEta() > mMaxEta) {
        return false;
      }
    }
    return true;
  }

 private:
  bool isBarrelTrack = false; // all barrel tracks must have either ITS or TPC contribution -> true if ITS || TPC track source condition is passed
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

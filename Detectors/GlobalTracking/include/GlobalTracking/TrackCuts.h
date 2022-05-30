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

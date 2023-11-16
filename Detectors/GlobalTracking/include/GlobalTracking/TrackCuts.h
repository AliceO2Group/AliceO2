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
  ////////////////////////  Selection setters  ///////////////////////
  /// ITS
  void setMinPtITSCut(float value) { mPtITSCut = value; }
  void setEtaITSCut(float value) { mEtaITSCut = value; }
  void setMinNClustersITS(float value) { mMinNClustersITS = value; }
  void setMaxChi2PerClusterITS(float value) { mMaxChi2PerClusterITS = value; }
  void setRequireHitsInITSLayers(int8_t minNRequiredHits, std::set<uint8_t> requiredLayers)
  {
    mRequiredITSHits.push_back(std::make_pair(minNRequiredHits, requiredLayers));
    LOG(info) << "Track selection, set require hits in ITS layers: " << static_cast<int>(minNRequiredHits);
  }
  /// TPC
  void setMinPtTPCCut(float value) { mPtTPCCut = value; }
  void setEtaTPCCut(float value) { mEtaTPCCut = value; }
  void setMinNTPCClustersCut(int32_t value) { mNTPCClustersCut = value; }
  void setMaxDCATPCCut(float value) { mDCATPCCut = value; }
  void setMaxDCATPCCutY(float value) { mDCATPCCutY = value; }
  /// ITS+TPC kinematics
  void setMinPtCut(float value) { mMinPt = value; }
  void setMaxPtCut(float value) { mMaxPt = value; }
  void setEtaCut(float valueMin, float valueMax)
  {
    mMinEta = valueMin;
    mMaxEta = valueMax;
  }

  //////////////////////////////// O2 ////////////////////////////////
  bool isSelected(GID trackIndex, o2::globaltracking::RecoContainer& data)
  {
    o2::track::TrackParCov trk;
    auto contributorsGID = data.getSingleDetectorRefs(trackIndex);
    auto src = trackIndex.getSource(); // make selections depending on track source
    // ITS tracks
    if (contributorsGID[GIndex::Source::ITS].isIndexSet()) { // ITS tracks selection
      const auto& itsTrk = data.getITSTrack(contributorsGID[GID::ITS]);
      int ITSnClusters = itsTrk.getNClusters();
      float ITSchi2 = itsTrk.getChi2();
      float itsChi2NCl = ITSnClusters != 0 ? ITSchi2 / (float)ITSnClusters : 0;
      uint8_t itsClusterMap = itsTrk.getPattern();
      if (itsTrk.getPt() <= mPtITSCut ||
          std::abs(itsTrk.getEta()) > mEtaITSCut || // TODO: define 2 different values for min and max (**)
          ITSnClusters <= mMinNClustersITS ||
          itsChi2NCl >= mMaxChi2PerClusterITS ||
          TrackMethods::FulfillsITSHitRequirements(itsClusterMap, mRequiredITSHits) == false) {
        return false;
      }
    }
    // TPC tracks
    if (contributorsGID[GIndex::Source::TPC].isIndexSet()) {
      const auto& tpcTrk = data.getTPCTrack(contributorsGID[GID::TPC]);
      math_utils::Point3D<float> v{}; // vertex not defined?!
      std::array<float, 2> dca;
      if (tpcTrk.getPt() < mPtTPCCut ||
          std::abs(tpcTrk.getEta()) > mEtaTPCCut || // TODO: define 2 different values for min and max (***)
          tpcTrk.getNClusters() < mNTPCClustersCut ||
          (!(const_cast<o2::tpc::TrackTPC&>(tpcTrk).propagateParamToDCA(v, mBz, &dca, mDCATPCCut)) ||
           std::abs(dca[0]) > mDCATPCCutY) ||
          std::hypot(dca[0], dca[1]) > mDCATPCCut) {
        return false;
      }
    }
    // ITS-TPC matched cuts
    // --> currently inactive in MatchITSTPCQC, since either GID::TPC or GID::ITS
    if (src == o2::dataformats::GlobalTrackID::ITSTPC ||
        src == o2::dataformats::GlobalTrackID::ITSTPCTRD ||
        src == o2::dataformats::GlobalTrackID::ITSTPCTOF ||
        src == o2::dataformats::GlobalTrackID::ITSTPCTRDTOF) { // track selection for barrel tracks (ITS-TPC matched)
      trk = data.getTrackParam(trackIndex);
      float trkEta = trk.getEta();
      if (trk.getPt() < mMinPt || trk.getPt() > mMaxPt || trkEta < mMinEta || trkEta > mMaxEta) {
        return false;
      }
    }
    return true;
  }

 private:
  //////////////////////  cut values   //////////////////////////////
  /// ITS track
  float mPtITSCut = 0.f;                                                // min pT for ITS track
  float mEtaITSCut = 1e10f;                                             // eta window for ITS track --> TODO: define 2 different values for min and max (**)
  int mMinNClustersITS{0};                                              // min number of ITS clusters
  float mMaxChi2PerClusterITS{1e10f};                                   // max its fit chi2 per ITS cluster
  std::vector<std::pair<int8_t, std::set<uint8_t>>> mRequiredITSHits{}; // vector of ITS requirements (minNRequiredHits in specific requiredLayers)
  /// TPC track
  float mPtTPCCut = 0.1f;        // min pT for TPC track
  float mEtaTPCCut = 1.4f;       // eta window for TPC track --> TODO: define 2 different values for min and max (***)
  int32_t mNTPCClustersCut = 60; // minimum number of TPC clusters for TPC track
  float mDCATPCCut = 100.f;      // max DCA 3D to PV for TPC track
  float mDCATPCCutY = 10.f;      // max DCA xy to PV for TPC track
  // ITS+TPC track kinematics
  float mMinPt{0.f}, mMaxPt{1e10f};      // range in pT
  float mMinEta{-1e10f}, mMaxEta{1e10f}; // range in eta

  float mBz = o2::base::Propagator::Instance()->getNominalBz();

  ClassDefNV(TrackCuts, 2);
};
} // namespace o2

#endif

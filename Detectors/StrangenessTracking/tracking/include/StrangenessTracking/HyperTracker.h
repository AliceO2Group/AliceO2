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

/// \file HyperTracker.h
/// \brief hypertracker
/// \author francesco.mazzaschi@cern.ch

#ifndef _ALICEO2_HYPER_TRACKER_
#define _ALICEO2_HYPER_TRACKER_
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITSBase/GeometryTGeo.h"
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsVertexing/DCAFitterN.h"

#include <TLorentzVector.h>
#include "TMath.h"

namespace o2
{
namespace tracking
{

class HyperTracker
{
 public:
  using PID = o2::track::PID;
  using TrackITS = o2::its::TrackITS;
  using ITSCluster = o2::BaseCluster<float>;
  using V0 = o2::dataformats::V0;
  using DCAFitter2 = o2::vertexing::DCAFitterN<2>;

  HyperTracker() = default;
  HyperTracker(const TrackITS& motherTrack, const V0& v0, const std::vector<ITSCluster>& motherClusters, o2::its::GeometryTGeo* gman, DCAFitter2& mFitterV0); //recompute V0 using hypertriton hypothesis
  HyperTracker(const TrackITS& motherTrack, const V0& v0, const std::vector<ITSCluster>& motherClusters, o2::its::GeometryTGeo* gman);

  double getMatchingChi2();
  double calcV0alpha(const V0& v0);
  bool process();
  bool propagateToClus(const ITSCluster& clus, o2::track::TrackParCov& track);
  int updateV0topology(const ITSCluster& clus, bool tryDaughter);
  bool recreateV0(const o2::track::TrackParCov& posTrack, const o2::track::TrackParCov& negTrack, const int posID, const int negID);
  V0& getV0() { return hypV0; };

  float getNclusMatching() const { return nClusMatching; }
  void setNclusMatching(float d) { nClusMatching = d; }

  float getMaxChi2() const { return mMaxChi2; }
  void setMaxChi2(float d) { mMaxChi2 = d; }

  float getBz() const { return mBz; }
  void setBz(float d) { mBz = d; }

 protected:
  TrackITS hyperTrack;                   // track of hypertriton mother
  V0 hypV0;                              // V0 of decay daughters
  std::vector<ITSCluster> hyperClusters; // clusters of hypertriton mother
  o2::its::GeometryTGeo* geomITS;        //geometry for ITS clusters
  DCAFitter2 mFitterV0;                  // optional DCA Fitter for recreating V0 with hypertriton mass hypothesis

  int nClusMatching; // number of cluster to be matched to V0
  float mMaxChi2 = 40;
  float mBz = -5;
};

} // namespace tracking
} // namespace o2

#endif //  _ALICEO2_HYPER_TRACKER_

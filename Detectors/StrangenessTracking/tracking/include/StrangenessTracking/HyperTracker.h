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
/// \brief
///

#ifndef _ALICEO2_HYPER_TRACKER_
#define _ALICEO2_HYPER_TRACKER_

#include <gsl/gsl>
#include <TLorentzVector.h>
#include "TMath.h"

#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITSBase/GeometryTGeo.h"
#include "ReconstructionDataFormats/Track.h"
#include "DataFormatsITSMFT/CompCluster.h"

#include "DetectorsVertexing/DCAFitterN.h"
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace strangeness_tracking
{

class HyperTracker
{
 public:
  using PID = o2::track::PID;
  using TrackITS = o2::its::TrackITS;
  using ITSCluster = o2::BaseCluster<float>;
  using V0 = o2::dataformats::V0;
  using DCAFitter2 = o2::vertexing::DCAFitterN<2>;
  using DCAFitter3 = o2::vertexing::DCAFitterN<3>;

  HyperTracker() = default;
  ~HyperTracker() = default;

  std::vector<V0>& getV0() { return mV0s; };
  std::vector<o2::track::TrackParCov>& getHyperTracks() { return mHyperTracks; };
  std::vector<int>& getITStrackRef() { return mITStrackRef; };

  float getMaxChi2() const { return mMaxChi2; }
  void setMaxChi2(float d) { mMaxChi2 = d; }
  float getBz() const { return mBz; }
  void setBz(float d) { mBz = d; }

  void setupFitters()
  {
    mFitterV0.setBz(mBz);
    mFitter3Body.setBz(mBz);
    mFitterV0.setUseAbsDCA(true);
    mFitter3Body.setUseAbsDCA(true);
  }

  bool loadData(gsl::span<const o2::its::TrackITS> InputITStracks, std::vector<ITSCluster>& InputITSclusters, gsl::span<const int> InputITSidxs, gsl::span<const V0> InputV0tracks, o2::its::GeometryTGeo* geomITS, float Bz);
  double calcV0alpha(const V0& v0);
  std::vector<ITSCluster> getTrackClusters(o2::its::TrackITS const& ITStrack);

  bool updateTrack(const ITSCluster& clus, o2::track::TrackParCov& track);

  void process();

  bool recreateV0(const o2::track::TrackParCov& posTrack, const o2::track::TrackParCov& negTrack, const int posID, const int negID);
  bool refitAllTracks();

 protected:
  gsl::span<const o2::its::TrackITS> mInputITStracks; // input ITS tracks
  std::vector<ITSCluster> mInputITSclusters;          // input ITS clusters
  gsl::span<const int> mInputITSidxs;                 // input ITS track-cluster indexes
  gsl::span<const V0> mInputV0tracks;                 // input V0 of decay daughters

  std::vector<o2::track::TrackParCov> mHyperTracks; // Final hypertrack
  std::vector<V0> mV0s;                             // V0 of decay daughters
  std::vector<int> mITStrackRef;                    // Ref to the ITS track

  DCAFitter2 mFitterV0;    // optional DCA Fitter for recreating V0 with hypertriton mass hypothesis
  DCAFitter3 mFitter3Body; // optional DCA Fitter for final 3 Body refit

  float mMaxChi2 = 40; // Maximum matching chi2
  float mBz = -5;      // Magnetic field

  o2::its::GeometryTGeo* mGeomITS; // ITS geometry
  V0 mV0;                          // V0 employed for the tracking

  ClassDefNV(HyperTracker, 1);
};

} // namespace strangeness_tracking
} // namespace o2

#endif //  _ALICEO2_HYPER_TRACKER_

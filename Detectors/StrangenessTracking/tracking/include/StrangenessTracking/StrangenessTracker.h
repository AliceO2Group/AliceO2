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

/// \file StrangenessTracker.h
/// \brief
///

#ifndef _ALICEO2_STRANGENESS_TRACKER_
#define _ALICEO2_STRANGENESS_TRACKER_

#include <gsl/gsl>
#include <TLorentzVector.h>
#include "TMath.h"
#include "StrangenessTracking/IndexTableUtils.h"
#include "ReconstructionDataFormats/PID.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"

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

struct ClusAttachments {

  std::array<unsigned int, 7> arr;
};

struct StrangeTrack {
  o2::track::TrackParCovF mMother;
  o2::track::TrackParCovF mDaughterFirst;
  o2::track::TrackParCovF mDaughterSecond;
  std::array<float, 3> decayVtx;

  float mMatchChi2;
  float mTopoChi2;
  int isCascade = false;
};

class StrangenessTracker
{
 public:
  enum kTopology { kMother = 1,
                   kFirstDaughter = 2,
                   kSecondDaughter = 3,
                   kThirdDaughter = 4,
                   kBachelor = 5 };

  using PID = o2::track::PID;
  using TrackITS = o2::its::TrackITS;
  using ITSCluster = o2::BaseCluster<float>;
  using V0 = o2::dataformats::V0;
  using Cascade = o2::dataformats::Cascade;

  using GIndex = o2::dataformats::VtxTrackIndex;
  using DCAFitter2 = o2::vertexing::DCAFitterN<2>;
  using DCAFitter3 = o2::vertexing::DCAFitterN<3>;

  StrangenessTracker() = default;
  ~StrangenessTracker() = default;

  void initialise();
  void process();

  std::vector<ClusAttachments>& getClusAttachments() { return mClusAttachments; };
  std::vector<StrangeTrack>& getStrangeTrackVec() { return mStrangeTrackVec; };
  std::vector<int>& getITStrackRefVec() { return mITStrackRefVec; };
  std::vector<int>& getDecayTrackRefVec() { return mDecayTrackRefVec; };

  float getMaxChi2() const { return mMaxChi2; }
  void setMaxChi2(float d) { mMaxChi2 = d; }
  float getBz() const { return mBz; }
  void setBz(float d) { mBz = d; }
  void setCorrType(const o2::base::PropagatorImpl<float>::MatCorrType& type) { mCorrType = type; }

  void setupFitters()
  {
    mFitterV0.setBz(mBz);
    mFitter3Body.setBz(mBz);
    mFitterV0.setUseAbsDCA(true);
    mFitter3Body.setUseAbsDCA(true);
  }

  bool loadData(gsl::span<const o2::its::TrackITS> InputITStracks, std::vector<ITSCluster>& InputITSclusters, gsl::span<const int> InputITSidxs, gsl::span<const V0> InputV0tracks, gsl::span<const Cascade> InputCascadeTracks, o2::its::GeometryTGeo* geomITS);
  double calcV0alpha(const V0& v0);
  std::vector<ITSCluster> getTrackClusters();
  float getMatchingChi2(o2::track::TrackParCovF, const TrackITS ITSTrack, ITSCluster matchingClus);
  bool recreateV0(const o2::track::TrackParCov& posTrack, const o2::track::TrackParCov& negTrack, const GIndex posID, const GIndex negID, V0& newV0);

  bool updateTrack(const ITSCluster& clus, o2::track::TrackParCov& track);
  bool matchDecayToITStrack(float decayR2, bool isCascade);



 protected:
  gsl::span<const o2::its::TrackITS> mInputITStracks; // input ITS tracks
  std::vector<int> mTracksIdxTable;                   // index table for ITS tracks
  std::vector<ITSCluster> mInputITSclusters;          // input ITS clusters
  gsl::span<const int> mInputITSidxs;                 // input ITS track-cluster indexes
  gsl::span<const V0> mInputV0tracks;                 // input V0 of decay daughters
  gsl::span<const Cascade> mInputCascadeTracks;       // input V0 of decay daughters

  std::vector<o2::its::TrackITS> mSortedITStracks; // sorted ITS tracks
  std::vector<int> mSortedITSindexes;              // indexes of sorted ITS tracks
  IndexTableUtils mUtils;                          // structure for computing eta/phi matching selections

  std::vector<StrangeTrack> mStrangeTrackVec;    // structure containing updated mother and daughter tracks
  std::vector<ClusAttachments> mClusAttachments; // # of attached tracks, 1 for mother, 2 for daughter
  std::vector<int> mITStrackRefVec;              // Ref to the ITS tracks
  std::vector<int> mDecayTrackRefVec;            // Ref to the Cascade and V0 files

  float mRadiusTol = 4.;     // Radius tolerance for matching V0s
  float mMinMotherClus = 3.; // minimum number of cluster to be attached to the mother
  float mMaxChi2 = 50;       // Maximum matching chi2
  float mBz = -5;            // Magnetic field

  DCAFitter2 mFitterV0;    // optional DCA Fitter for recreating V0 with hypertriton mass hypothesis
  DCAFitter3 mFitter3Body; // optional DCA Fitter for final 3 Body refit

  o2::base::PropagatorImpl<float>::MatCorrType mCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE; // use mat correction
  o2::its::GeometryTGeo* mGeomITS;                                                                                       // ITS geometry
  StrangeTrack mStrangeTrack;
  ClusAttachments mStructClus;
  o2::its::TrackITS mITStrack;

  ClassDefNV(StrangenessTracker, 1);
};

} // namespace strangeness_tracking
} // namespace o2

#endif //  _ALICEO2_STRANGENESS_TRACKER_

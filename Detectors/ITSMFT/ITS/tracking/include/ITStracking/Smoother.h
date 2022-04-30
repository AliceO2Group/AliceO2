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
//
/// \file Smoother.h
/// \brief Class to handle Kalman smoothing for ITS tracking.
///         Its instance stores the state of the track to the level we want to smooth to avoid multiple re-propagations when testing different clusters.
///

#include "ReconstructionDataFormats/Track.h"
#include "DataFormatsITS/TrackITS.h"
#include "DetectorsBase/Propagator.h"
#include "ITStracking/ROframe.h"

namespace o2
{
namespace its
{

template <unsigned int D>
class Smoother
{
 public:
  Smoother(TrackITSExt& track, size_t layer, const ROframe& event, float bZ, o2::base::PropagatorF::MatCorrType corr);
  ~Smoother();

  bool isValidInit() const
  {
    return mInitStatus;
  }
  bool testCluster(const int clusterId, const ROframe& event);
  bool getSmoothedTrack();
  float getChi2() const { return mBestChi2; }
  float getLastChi2() const { return mLastChi2; }

 private:
  float computeSmoothedPredictedChi2(const o2::track::TrackParCov& outwTrack,
                                     const o2::track::TrackParCov& inwTrack,
                                     const std::array<float, 2>& cls,
                                     const std::array<float, 3>& clCov);
  bool smoothTrack();

 private:
  size_t mLayerToSmooth;                    // Layer to compute smoothing optimization
  float mBz;                                // Magnetic field along Z
  bool mInitStatus;                         // State after the initialization
  o2::base::PropagatorF::MatCorrType mCorr; // Type of correction to use
  TrackITSExt mInwardsTrack;                // outwards track: from innermost cluster to outermost
  TrackITSExt mOutwardsTrack;               // inwards track: from outermost cluster to innermost
  float mBestChi2;                          // Best value of local smoothed chi2
  float mLastChi2 = 1e8;                    // Latest computed chi2
};
} // namespace its
} // namespace o2

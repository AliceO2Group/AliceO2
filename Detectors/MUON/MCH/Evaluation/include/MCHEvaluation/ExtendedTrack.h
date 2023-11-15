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

#ifndef O2_MCH_EVALUATION_EXTENDED_TRACK_H__
#define O2_MCH_EVALUATION_EXTENDED_TRACK_H__

#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "MCHTracking/Track.h"
#include "Math/Vector4D.h"
#include <gsl/span>
#include <iostream>
#include <vector>

namespace o2::mch
{
namespace eval
{

/** ExtendedTrack is an extension of a TrackMCH.
 *
 * An ExtendedTrack is a TrackMCH (standalone MCH) that has been
 * extrapolated to (a) vertex. It also contains plain copies of the
 * clusters used to define the original TrackMCH.
 *
 */
class ExtendedTrack
{
 public:
  /** create an ExtendedTrack from a TrackMCH and its clusters.
   *
   * The given TrackMCH will be extrapolated to the given vertex {x,y,z}.
   * Throw an exception if the track fitting fails
   */
  ExtendedTrack(const TrackMCH& track,
                gsl::span<const Cluster>& clusters,
                double x, double y, double z);

  bool operator==(const ExtendedTrack& track) const;

  bool isMatching(const ExtendedTrack& track) const;

  const std::vector<Cluster>& getClusters() const { return mClusters; }

  std::string asString() const;

  bool hasMatchFound() const { return mHasMatchFound; }
  bool hasMatchIdentical() const { return mHasMatchIdentical; }

  void setMatchFound(bool val = true) { mHasMatchFound = val; }
  void setMatchIdentical(bool val = true) { mHasMatchIdentical = val; }

  const TrackParam& param() const;
  const Track& track() const;

  double getDCA() const { return mDCA; }
  double getRabs() const { return mRabs; }
  double getCharge() const { return param().getCharge(); }

  const ROOT::Math::PxPyPzMVector& P() const { return mMomentum4D; }

  double getNormalizedChi2() const;

 private:
  void extrapToVertex(double x, double y, double z);

 private:
  Track mTrack{};
  std::vector<Cluster> mClusters{};
  ROOT::Math::PxPyPzMVector mMomentum4D{};
  bool mHasMatchFound;
  bool mHasMatchIdentical;
  double mDCA{0.};
  double mRabs{0.};
  static constexpr double sChi2Max{2. * 4. * 4.};
};

std::ostream& operator<<(std::ostream& out, const ExtendedTrack& track);

/** tracks are considered identical when all their clusters match within chi2Max
 */
bool areEqual(const ExtendedTrack& t1, const ExtendedTrack& t2, double chi2Max);

/* Try to match this track with the given track.
 *
 * Matching conditions:
 * - more than 50% of clusters from one of the two tracks matched with
 *   clusters from the other
 * - at least 1 cluster matched before and 1 cluster matched after the dipole
 */
bool areMatching(const ExtendedTrack& t1, const ExtendedTrack& t2, double chi2Max);

} // namespace eval
}; // namespace o2::mch

#endif

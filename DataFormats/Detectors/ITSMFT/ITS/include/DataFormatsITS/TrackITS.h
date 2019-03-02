// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackITS.h
/// \brief Definition of the ITS track
/// \author iouri.belikov@cern.ch

#ifndef ALICEO2_ITS_TRACKITS_H
#define ALICEO2_ITS_TRACKITS_H

#include <vector>

#include "ReconstructionDataFormats/Track.h"

namespace o2
{
namespace itsmft
{
class Cluster;
}

namespace ITS
{
class TrackITS : public o2::track::TrackParCov
{
  using Cluster = o2::itsmft::Cluster;

 public:
  using o2::track::TrackParCov::TrackParCov; // inherit base constructors
  static constexpr int MaxClusters = 7;

  TrackITS() = default;
  TrackITS(const TrackITS& t) = default;
  TrackITS(o2::track::TrackParCov&& parcov) : TrackParCov{ parcov } {}
  TrackITS(o2::track::TrackParCov&& parCov, short ncl, float mass, float chi2, std::uint32_t rof, o2::track::TrackParCov&& outer, std::array<int, MaxClusters> cls) : o2::track::TrackParCov{ parCov }, mNClusters{ ncl }, mMass{ mass }, mChi2{ chi2 }, mROFrame{ rof }, mParamOut{ outer }, mIndex{ cls } {}
  TrackITS& operator=(const TrackITS& tr) = default;
  ~TrackITS() = default;

  // These functions must be provided
  bool propagate(float alpha, float x, float bz);
  bool update(const Cluster& c, float chi2, int idx);

  // Other functions
  float getChi2() const { return mChi2; }
  int getNumberOfClusters() const { return mNClusters; }
  int getClusterIndex(int i) const { return mIndex[i]; }
  bool operator<(const TrackITS& o) const;
  void getImpactParams(float x, float y, float z, float bz, float ip[2]) const;
  // bool getPhiZat(float r,float &phi,float &z) const;

  void setClusterIndex(int layer, int index);
  void setExternalClusterIndex(int layer, int idx, bool newCluster = false);
  void resetClusters();

  void setChi2(float chi2) { mChi2 = chi2; }

  std::uint32_t getROFrame() const { return mROFrame; }
  void setROFrame(std::uint32_t f) { mROFrame = f; }
  bool isBetter(const TrackITS& best, float maxChi2) const;

  o2::track::TrackParCov& getParamOut() { return mParamOut; }
  const o2::track::TrackParCov& getParamOut() const { return mParamOut; }

 private:
  short mNClusters = 0;
  float mMass = 0.14;                             ///< Assumed mass for this track
  float mChi2 = 0.;                               ///< Chi2 for this track
  std::uint32_t mROFrame = 0;                     ///< RO Frame
  o2::track::TrackParCov mParamOut;               /// parameter at largest radius
  std::array<int, MaxClusters> mIndex = { -1 };   ///< Indices of associated clusters

  ClassDefNV(TrackITS, 2)
};
}
}
#endif /* ALICEO2_ITS_TRACKITS_H */

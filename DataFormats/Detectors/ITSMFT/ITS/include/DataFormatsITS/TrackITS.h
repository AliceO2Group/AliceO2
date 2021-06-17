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
#include "CommonDataFormat/RangeReference.h"

namespace o2
{
namespace itsmft
{
class Cluster;
}

namespace its
{

class TrackITS : public o2::track::TrackParCov
{
  using Cluster = o2::itsmft::Cluster;
  using ClusRefs = o2::dataformats::RangeRefComp<4>;

 public:
  using o2::track::TrackParCov::TrackParCov; // inherit base constructors
  static constexpr int MaxClusters = 16;

  TrackITS() = default;
  TrackITS(const TrackITS& t) = default;
  TrackITS(const o2::track::TrackParCov& parcov) : o2::track::TrackParCov{parcov} {}
  TrackITS(const o2::track::TrackParCov& parCov, float chi2, const o2::track::TrackParCov& outer)
    : o2::track::TrackParCov{parCov}, mParamOut{outer}, mChi2{chi2} {}
  TrackITS& operator=(const TrackITS& tr) = default;
  ~TrackITS() = default;

  // These functions must be provided
  bool propagate(float alpha, float x, float bz);
  bool update(const Cluster& c, float chi2);

  // Other functions
  float getChi2() const { return mChi2; }
  int getNClusters() const { return mClusRef.getEntries(); }
  int getNumberOfClusters() const { return getNClusters(); }
  int getFirstClusterEntry() const { return mClusRef.getFirstEntry(); }
  int getClusterEntry(int i) const { return getFirstClusterEntry() + i; }
  void shiftFirstClusterEntry(int bias)
  {
    mClusRef.setFirstEntry(mClusRef.getFirstEntry() + bias);
  }
  void setFirstClusterEntry(int offs)
  {
    mClusRef.setFirstEntry(offs);
  }
  void setNumberOfClusters(int n)
  {
    mClusRef.setEntries(n);
  }
  bool operator<(const TrackITS& o) const;
  void getImpactParams(float x, float y, float z, float bz, float ip[2]) const;
  // bool getPhiZat(float r,float &phi,float &z) const;

  void setClusterRefs(int firstEntry, int n)
  {
    mClusRef.set(firstEntry, n);
  }

  const ClusRefs& getClusterRefs() const { return mClusRef; }
  ClusRefs& getClusterRefs() { return mClusRef; }

  void setChi2(float chi2) { mChi2 = chi2; }

  bool isBetter(const TrackITS& best, float maxChi2) const;

  o2::track::TrackParCov& getParamOut() { return mParamOut; }
  const o2::track::TrackParCov& getParamOut() const { return mParamOut; }

  void setPattern(uint32_t p) { mPattern = p; }
  int getPattern() const { return mPattern; }
  bool hasHitOnLayer(int i) { return mPattern & (0x1 << i); }
  bool isFakeOnLayer(int i) { return !(mPattern & (0x1 << (16 + i))); }

 private:
  o2::track::TrackParCov mParamOut; ///< parameter at largest radius
  ClusRefs mClusRef;                ///< references on clusters
  float mChi2 = 0.;                 ///< Chi2 for this track
  uint32_t mPattern = 0;            ///< layers pattern

  ClassDefNV(TrackITS, 5);
};

class TrackITSExt : public TrackITS
{
  ///< heavy version of TrackITS, with clusters embedded
 public:
  static constexpr int MaxClusters = 16; /// Prepare for overlaps and new detector configurations
  using TrackITS::TrackITS;              // inherit base constructors

  TrackITSExt(o2::track::TrackParCov&& parCov, short ncl, float chi2,
              o2::track::TrackParCov&& outer, std::array<int, MaxClusters> cls)
    : TrackITS(parCov, chi2, outer), mIndex{cls}
  {
    setNumberOfClusters(ncl);
  }

  TrackITSExt(o2::track::TrackParCov& parCov, short ncl, float chi2, std::uint32_t rof,
              o2::track::TrackParCov& outer, std::array<int, MaxClusters> cls)
    : TrackITS(parCov, chi2, outer), mIndex{cls}
  {
    setNumberOfClusters(ncl);
  }

  void setClusterIndex(int l, int i)
  {
    int ncl = getNumberOfClusters();
    mIndex[ncl++] = (l << 28) + i;
    getClusterRefs().setEntries(ncl);
  }

  int getClusterIndex(int lr) const { return mIndex[lr]; }

  void setExternalClusterIndex(int layer, int idx, bool newCluster = false)
  {
    if (newCluster) {
      getClusterRefs().setEntries(getNumberOfClusters() + 1);
      uint32_t pattern = getPattern();
      pattern |= 0x1 << layer;
      setPattern(pattern);
    }
    mIndex[layer] = idx;
  }

  std::array<int, MaxClusters>& getClusterIndexes()
  {
    return mIndex;
  }

 private:
  std::array<int, MaxClusters> mIndex = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}; ///< Indices of associated clusters
  ClassDefNV(TrackITSExt, 2);
};
} // namespace its
} // namespace o2
#endif /* ALICEO2_ITS_TRACKITS_H */

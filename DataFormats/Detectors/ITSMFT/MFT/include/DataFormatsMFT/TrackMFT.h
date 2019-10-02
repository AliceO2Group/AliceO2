// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Track.h
/// \brief Definition of the MFT track
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 8, 2018

#ifndef ALICEO2_MFT_TRACKMFT_H
#define ALICEO2_MFT_TRACKMFT_H

#include <vector>

#include "ReconstructionDataFormats/Track.h"
#include "CommonDataFormat/RangeReference.h"

namespace o2
{

namespace itsmft
{
class Cluster;
}

namespace mft
{
class TrackMFT : public o2::track::TrackParCov
{
  using Cluster = o2::itsmft::Cluster;
  using ClusRefs = o2::dataformats::RangeRefComp<4>;

 public:
  using o2::track::TrackParCov::TrackParCov;

  TrackMFT() = default;
  TrackMFT(const TrackMFT& t) = default;
  TrackMFT(const o2::track::TrackParCov& parcov) : TrackParCov{parcov} {}
  TrackMFT(const o2::track::TrackParCov& parCov, float chi2, std::uint32_t rof, const o2::track::TrackParCov& outer)
    : o2::track::TrackParCov{parCov}, mChi2{chi2}, mROFrame{rof}, mParamOut{outer} {}
  TrackMFT& operator=(const TrackMFT& tr) = default;
  ~TrackMFT() = default;

  // Other functions
  float getChi2() const { return mChi2; }
  int getNumberOfClusters() const { return mClusRef.getEntries(); }
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
  void setClusterRefs(int firstEntry, int n)
  {
    mClusRef.set(firstEntry, n);
  }

  const ClusRefs& getClusterRefs() const { return mClusRef; }
  ClusRefs& getClusterRefs() { return mClusRef; }

  void setChi2(float chi2) { mChi2 = chi2; }

  std::uint32_t getROFrame() const { return mROFrame; }
  void setROFrame(std::uint32_t f) { mROFrame = f; }

  o2::track::TrackParCov& getParamOut() { return mParamOut; }
  const o2::track::TrackParCov& getParamOut() const { return mParamOut; }

 private:
  float mChi2 = 0.;                 ///< Chi2 for this track
  std::uint32_t mROFrame = 0;       ///< RO Frame
  o2::track::TrackParCov mParamOut; ///< parameter at largest radius
  ClusRefs mClusRef;                ///< references on clusters

  ClassDefNV(TrackMFT, 1);
};

class TrackMFTExt : public TrackMFT
{
  ///< heavy version of TrackMFT, with clusters embedded
 public:
  static constexpr int MaxClusters = 10;
  using TrackMFT::TrackMFT; // inherit base constructors

  TrackMFTExt(o2::track::TrackParCov&& parCov, short ncl, float chi2, std::uint32_t rof,
              o2::track::TrackParCov&& outer, std::array<int, MaxClusters> cls)
    : TrackMFT(parCov, chi2, rof, outer), mIndex{cls}
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
    }
    mIndex[layer] = idx;
  }

 private:
  std::array<int, MaxClusters> mIndex = {-1}; ///< Indices of associated clusters
  ClassDefNV(TrackMFTExt, 1);
};
} // namespace mft
} // namespace o2

#endif

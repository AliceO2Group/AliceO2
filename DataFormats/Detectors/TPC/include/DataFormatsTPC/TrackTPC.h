// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_TRACKTPC
#define ALICEO2_TPC_TRACKTPC

#include "ReconstructionDataFormats/Track.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/Defs.h"
#include "DataFormatsTPC/dEdxInfo.h"
#include <gsl/span>

namespace o2
{
namespace tpc
{
/// \class TrackTPC
/// This is the definition of the TPC Track Object

using TPCClRefElem = uint32_t;

class TrackTPC : public o2::track::TrackParCov
{
  using ClusRef = o2::dataformats::RangeReference<uint32_t, uint16_t>;

 public:
  enum Flags : unsigned short {
    HasASideClusters = 0x1 << 0,                                ///< track has clusters on A side
    HasCSideClusters = 0x1 << 1,                                ///< track has clusters on C side
    HasBothSidesClusters = HasASideClusters | HasCSideClusters, // track has clusters on both sides
    FullMask = 0xffff
  };

  using o2::track::TrackParCov::TrackParCov; // inherit

  /// Default constructor
  TrackTPC() = default;

  /// Destructor
  ~TrackTPC() = default;

  unsigned short getFlags() const { return mFlags; }
  unsigned short getClustersSideInfo() const { return mFlags & HasBothSidesClusters; }
  bool hasASideClusters() const { return mFlags & HasASideClusters; }
  bool hasCSideClusters() const { return mFlags & HasCSideClusters; }
  bool hasBothSidesClusters() const { return (mFlags & (HasASideClusters | HasCSideClusters)) == (HasASideClusters | HasCSideClusters); }
  bool hasASideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasASideClusters; }
  bool hasCSideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasCSideClusters; }

  void setHasASideClusters() { mFlags |= HasASideClusters; }
  void setHasCSideClusters() { mFlags |= HasCSideClusters; }

  float getTime0() const { return mTime0; }         ///< Reference time of the track, i.e. t-bins of a primary track with eta=0.
  short getDeltaTBwd() const { return mDeltaTBwd; } ///< max possible decrement to getTimeVertex
  short getDeltaTFwd() const { return mDeltaTFwd; } ///< max possible increment to getTimeVertex
  void setDeltaTBwd(short t) { mDeltaTBwd = t; }    ///< set max possible decrement to getTimeVertex
  void setDeltaTFwd(short t) { mDeltaTFwd = t; }    ///< set max possible increment to getTimeVertex

  float getChi2() const { return mChi2; }
  const o2::track::TrackParCov& getOuterParam() const { return mOuterParam; }
  void setTime0(float v) { mTime0 = v; }
  void setChi2(float v) { mChi2 = v; }
  void setOuterParam(o2::track::TrackParCov&& v) { mOuterParam = v; }
  const ClusRef& getClusterRef() const { return mClustersReference; }
  void shiftFirstClusterRef(int dif) { mClustersReference.setFirstEntry(dif + mClustersReference.getFirstEntry()); }
  int getNClusters() const { return mClustersReference.getEntries(); }
  int getNClusterReferences() const { return getNClusters(); }
  void setClusterRef(uint32_t entry, uint16_t ncl) { mClustersReference.set(entry, ncl); }

  void getClusterReference(gsl::span<const o2::tpc::TPCClRefElem> clinfo, int nCluster,
                           uint8_t& sectorIndex, uint8_t& rowIndex, uint32_t& clusterIndex) const
  {
    // data for given tracks starts at clinfo[ mClustersReference.getFirstEntry() ],
    // 1st mClustersReference.getEntries() cluster indices are stored as uint32_t
    // then sector indices as uint8_t, then row indices ar uin8_t

    //    const uint32_t* clIndArr = &clinfo[ mClustersReference.getFirstEntry() ];
    const uint32_t* clIndArr = reinterpret_cast<const uint32_t*>(&clinfo[mClustersReference.getFirstEntry()]); // TODO remove this trick
    clusterIndex = clIndArr[nCluster];
    const uint8_t* srIndexArr = reinterpret_cast<const uint8_t*>(clIndArr + mClustersReference.getEntries());
    sectorIndex = srIndexArr[nCluster];
    rowIndex = srIndexArr[nCluster + mClustersReference.getEntries()];
  }

  const o2::tpc::ClusterNative& getCluster(gsl::span<const o2::tpc::TPCClRefElem> clinfo, int nCluster,
                                           const o2::tpc::ClusterNativeAccess& clusters, uint8_t& sectorIndex, uint8_t& rowIndex) const
  {
    uint32_t clusterIndex;
    getClusterReference(clinfo, nCluster, sectorIndex, rowIndex, clusterIndex);
    return (clusters.clusters[sectorIndex][rowIndex][clusterIndex]);
  }

  const o2::tpc::ClusterNative& getCluster(gsl::span<const o2::tpc::TPCClRefElem> clinfo, int nCluster,
                                           const o2::tpc::ClusterNativeAccess& clusters) const
  {
    uint8_t sectorIndex, rowIndex;
    return (getCluster(clinfo, nCluster, clusters, sectorIndex, rowIndex));
  }

  const dEdxInfo& getdEdx() const { return mdEdx; }
  void setdEdx(const dEdxInfo& v) { mdEdx = v; }

 private:
  float mTime0 = 0.f;                 ///< Reference Z of the track assumed for the vertex, scaled with pseudo
                                      ///< VDrift and reference timeframe length, unless it was moved to be on the
                                      ///< side of TPC compatible with edge clusters sides.
  short mDeltaTFwd = 0;               ///< max possible increment to track time
  short mDeltaTBwd = 0;               ///< max possible decrement to track time
  short mFlags = 0;                   ///< various flags, see Flags enum
  float mChi2 = 0.f;                  // Chi2 of the track
  o2::track::TrackParCov mOuterParam; // Track parameters at outer end of TPC.
  dEdxInfo mdEdx;                     // dEdx Information
  ClusRef mClustersReference;         // reference to externale cluster indices

  ClassDefNV(TrackTPC, 3);
};

} // namespace tpc
} // namespace o2

#endif

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

#include "GPUCommonDef.h"
#include "ReconstructionDataFormats/Track.h"
#include "CommonDataFormat/RangeReference.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "DataFormatsTPC/dEdxInfo.h"

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
  GPUdDefault() TrackTPC() = default;

  /// Destructor
  GPUdDefault() ~TrackTPC() = default;

  GPUd() unsigned short getFlags() const { return mFlags; }
  GPUd() unsigned short getClustersSideInfo() const { return mFlags & HasBothSidesClusters; }
  GPUd() bool hasASideClusters() const { return mFlags & HasASideClusters; }
  GPUd() bool hasCSideClusters() const { return mFlags & HasCSideClusters; }
  GPUd() bool hasBothSidesClusters() const { return (mFlags & (HasASideClusters | HasCSideClusters)) == (HasASideClusters | HasCSideClusters); }
  GPUd() bool hasASideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasASideClusters; }
  GPUd() bool hasCSideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasCSideClusters; }

  GPUd() void setHasASideClusters() { mFlags |= HasASideClusters; }
  GPUd() void setHasCSideClusters() { mFlags |= HasCSideClusters; }

  GPUd() float getTime0() const { return mTime0; }         ///< Reference time of the track, i.e. t-bins of a primary track with eta=0.
  GPUd() float getDeltaTBwd() const { return mDeltaTBwd; } ///< max possible decrement to getTimeVertex
  GPUd() float getDeltaTFwd() const { return mDeltaTFwd; } ///< max possible increment to getTimeVertex
  GPUd() void setDeltaTBwd(float t) { mDeltaTBwd = t; }    ///< set max possible decrement to getTimeVertex
  GPUd() void setDeltaTFwd(float t) { mDeltaTFwd = t; }    ///< set max possible increment to getTimeVertex

  GPUd() float getChi2() const { return mChi2; }
  GPUd() const o2::track::TrackParCov& getOuterParam() const { return mOuterParam; }
  GPUd() const o2::track::TrackParCov& getParamOut() const { return mOuterParam; } // to have method with same name as other tracks
  GPUd() void setTime0(float v) { mTime0 = v; }
  GPUd() void setChi2(float v) { mChi2 = v; }
  GPUd() void setOuterParam(o2::track::TrackParCov&& v) { mOuterParam = v; }
  GPUd() void setParamOut(o2::track::TrackParCov&& v) { mOuterParam = v; } // to have method with same name as other tracks
  GPUd() const ClusRef& getClusterRef() const { return mClustersReference; }
  GPUd() void shiftFirstClusterRef(int dif) { mClustersReference.setFirstEntry(dif + mClustersReference.getFirstEntry()); }
  GPUd() int getNClusters() const { return mClustersReference.getEntries(); }
  GPUd() int getNClusterReferences() const { return getNClusters(); }
  GPUd() void setClusterRef(uint32_t entry, uint16_t ncl) { mClustersReference.set(entry, ncl); }

  template <class T>
  GPUd() static inline void getClusterReference(T& clinfo, int nCluster,
                                                uint8_t& sectorIndex, uint8_t& rowIndex, uint32_t& clusterIndex, const ClusRef& ref)
  {
    // data for given tracks starts at clinfo[ ref.getFirstEntry() ],
    // 1st ref.getEntries() cluster indices are stored as uint32_t
    // then sector indices as uint8_t, then row indices ar uin8_t

    //    const uint32_t* clIndArr = &clinfo[ ref.getFirstEntry() ];
    const uint32_t* clIndArr = reinterpret_cast<const uint32_t*>(&clinfo[ref.getFirstEntry()]); // TODO remove this trick
    clusterIndex = clIndArr[nCluster];
    const uint8_t* srIndexArr = reinterpret_cast<const uint8_t*>(clIndArr + ref.getEntries());
    sectorIndex = srIndexArr[nCluster];
    rowIndex = srIndexArr[nCluster + ref.getEntries()];
  }

  template <class T>
  GPUd() inline void getClusterReference(T& clinfo, int nCluster,
                                         uint8_t& sectorIndex, uint8_t& rowIndex, uint32_t& clusterIndex) const
  {
    getClusterReference<T>(clinfo, nCluster, sectorIndex, rowIndex, clusterIndex, mClustersReference);
  }

  template <class T>
  GPUd() static inline const o2::tpc::ClusterNative& getCluster(T& clinfo, int nCluster,
                                                                const o2::tpc::ClusterNativeAccess& clusters, uint8_t& sectorIndex, uint8_t& rowIndex, const ClusRef& ref)
  {
    uint32_t clusterIndex;
    getClusterReference<T>(clinfo, nCluster, sectorIndex, rowIndex, clusterIndex, ref);
    return (clusters.clusters[sectorIndex][rowIndex][clusterIndex]);
  }

  template <class T>
  GPUd() inline const o2::tpc::ClusterNative& getCluster(T& clinfo, int nCluster,
                                                         const o2::tpc::ClusterNativeAccess& clusters, uint8_t& sectorIndex, uint8_t& rowIndex) const
  {
    return getCluster<T>(clinfo, nCluster, clusters, sectorIndex, rowIndex, mClustersReference);
  }

  template <class T>
  GPUd() inline const o2::tpc::ClusterNative& getCluster(T& clinfo, int nCluster,
                                                         const o2::tpc::ClusterNativeAccess& clusters) const
  {
    uint8_t sectorIndex, rowIndex;
    return (getCluster<T>(clinfo, nCluster, clusters, sectorIndex, rowIndex));
  }

  GPUd() const dEdxInfo& getdEdx() const { return mdEdx; }
  GPUd() void setdEdx(const dEdxInfo& v) { mdEdx = v; }

 private:
  float mTime0 = 0.f;                 ///< Assumed time of the vertex that created the track in TPC time bins, 0 for triggered data
  float mDeltaTFwd = 0;               ///< max possible increment to mTime0
  float mDeltaTBwd = 0;               ///< max possible decrement to mTime0
  short mFlags = 0;                   ///< various flags, see Flags enum
  float mChi2 = 0.f;                  // Chi2 of the track
  o2::track::TrackParCov mOuterParam; // Track parameters at outer end of TPC.
  dEdxInfo mdEdx;                     // dEdx Information
  ClusRef mClustersReference;         // reference to externale cluster indices

  ClassDefNV(TrackTPC, 4);
};

} // namespace tpc
} // namespace o2

#endif

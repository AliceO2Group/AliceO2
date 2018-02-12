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

#include "DataFormatsTPC/ClusterNative.h"
#include "TPCBase/Defs.h"
#include "DataFormatsTPC/Cluster.h"

namespace o2
{
namespace TPC
{
/// \class TrackTPC
/// This is the definition of the TPC Track Object

class TrackTPC : public o2::track::TrackParCov
{
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

  /// Add a single cluster to the track
  void addCluster(const Cluster& c);

  /// Add an array/vector of clusters to the track; ClusterType needs to inherit from o2::TPC::Cluster
  template <typename ClusterType>
  void addClusterArray(std::vector<ClusterType>* arr);

  /// Get the clusters which are associated with the track
  /// \return clusters of the track as a std::vector
  void getClusterVector(std::vector<Cluster>& clVec) const { clVec = mClusterVector; }
  /// Get the truncated mean energy loss of the track
  /// \param low low end of truncation
  /// \param high high end of truncation
  /// \param type 0 for Qmax, 1 for Q
  /// \param removeRows option to remove certain rows from the dEdx calculation
  /// \param nclPID pass any pointer to have the number of used clusters written to it
  /// \return mean energy loss
  float getTruncatedMean(float low = 0.05, float high = 0.7, int type = 1, int removeRows = 0,
                         int* nclPID = nullptr) const;

  unsigned short getFlags() const { return mFlags; }
  unsigned short getClustersSideInfo() const { return mFlags & HasBothSidesClusters; }
  bool hasASideClusters() const { return mFlags & HasASideClusters; }
  bool hasCSideClusters() const { return mFlags & HasCSideClusters; }
  bool hasBothSidesClusters() const { return mFlags & (HasASideClusters | HasCSideClusters); }
  bool hasASideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasASideClusters; }
  bool hasCSideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasCSideClusters; }

  void setHasASideClusters() { mFlags |= HasASideClusters; }
  void setHasCSideClusters() { mFlags |= HasCSideClusters; }

  float getTime0() const { return mTime0; } ///< Reference time of the track, i.e. t-bins of a primary track with eta=0.
  float getTimeVertex(float vDrift) const;  ///< Abs time of the vertex assumed for the track: time0-drift time.
  short getDeltaTBwd() const { return mDeltaTBwd; } ///< max possible decrement to getTimeVertex
  short getDeltaTFwd() const { return mDeltaTFwd; } ///< max possible increment to getTimeVertex
  void setDeltaTBwd(short t) { mDeltaTBwd = t; }    ///< set max possible decrement to getTimeVertex
  void setDeltaTFwd(short t) { mDeltaTFwd = t; }    ///< set max possible increment to getTimeVertex

  float getChi2() const { return mChi2; }
  const o2::track::TrackParCov& getOuterParam() const { return mOuterParam; }
  void setTime0(float v) { mTime0 = v; }
  void setChi2(float v) { mChi2 = v; }
  void setOuterParam(o2::track::TrackParCov&& v) { mOuterParam = v; }
  void resetClusterReferences(int nClusters);
  int getNClusterReferences() const { return mNClusters; }
  void setClusterReference(int nCluster, uint8_t sectorIndex, uint8_t rowIndex, uint32_t clusterIndex)
  {
    mClusterReferences[nCluster] = clusterIndex;
    reinterpret_cast<uint8_t*>(mClusterReferences.data())[4 * mNClusters + nCluster] = sectorIndex;
    reinterpret_cast<uint8_t*>(mClusterReferences.data())[5 * mNClusters + nCluster] = rowIndex;
  }
  void getClusterReference(int nCluster, uint8_t& sectorIndex, uint8_t& rowIndex, uint32_t& clusterIndex) const
  {
    clusterIndex = mClusterReferences[nCluster];
    sectorIndex = reinterpret_cast<const uint8_t*>(mClusterReferences.data())[4 * mNClusters + nCluster];
    rowIndex = reinterpret_cast<const uint8_t*>(mClusterReferences.data())[5 * mNClusters + nCluster];
  }
  const o2::TPC::ClusterNative& getCluster(int nCluster, const o2::TPC::ClusterNativeAccessFullTPC& clusters,
                                           uint8_t& sectorIndex, uint8_t& rowIndex) const
  {
    uint32_t clusterIndex;
    getClusterReference(nCluster, sectorIndex, rowIndex, clusterIndex);
    return (clusters.clusters[sectorIndex][rowIndex][clusterIndex]);
  }
  const o2::TPC::ClusterNative& getCluster(int nCluster, const o2::TPC::ClusterNativeAccessFullTPC& clusters) const
  {
    uint8_t sectorIndex, rowIndex;
    return (getCluster(nCluster, clusters, sectorIndex, rowIndex));
  }

 private:
  std::vector<Cluster> mClusterVector;
  float mTime0 = 0.f;                 ///< Reference Z of the track assumed for the vertex, scaled with pseudo
                                      ///< VDrift and reference timeframe length, unless it was moved to be on the
                                      ///< side of TPC compatible with edge clusters sides.
  short mDeltaTFwd = 0;               ///< max possible increment to track time
  short mDeltaTBwd = 0;               ///< max possible decrement to track time
  short mNClusters = 0;               ///< number of clusters attached
  short mFlags = 0;                   ///< various flags, see Flags enum
  float mChi2 = 0.f;                  // Chi2 of the track
  o2::track::TrackParCov mOuterParam; // Track parameters at outer end of TPC.

  // New structure to store cluster references
  std::vector<uint32_t> mClusterReferences;

  ClassDefNV(TrackTPC, 2); // RS TODO set to 1
};

inline void TrackTPC::addCluster(const Cluster& c) { mClusterVector.push_back(c); }
template <typename ClusterType>
inline void TrackTPC::addClusterArray(std::vector<ClusterType>* arr)
{
  static_assert(std::is_base_of<o2::TPC::Cluster, ClusterType>::value,
                "ClusterType needs to inherit from o2::TPC::Cluster");
  for (auto clusterObject : *arr) {
    addCluster(clusterObject);
  }
}
}
}

#endif

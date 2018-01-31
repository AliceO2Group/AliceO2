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
#include "TPCReconstruction/Cluster.h"

namespace o2
{
namespace TPC
{
/// \class TrackTPC
/// This is the definition of the TPC Track Object

class TrackTPC : public o2::track::TrackParCov
{
 public:
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

  float getTime0() const { return mTime0; } // Reference time of the track, i.e. timebins of a primary track with eta=0.
  float getTimeVertex(
    float vDrift) const; // Absolute time of the vertex assumed for the track, shifted by time0 by one drift time.
  float getLastClusterZ() const { return mLastClusterZ; }
  Side getSide() const { return (Side)mSide; }
  float getChi2() const { return mChi2; }
  const o2::track::TrackParCov& getOuterParam() const { return mOuterParam; }
  void setTime0(float v) { mTime0 = v; }
  void setLastClusterZ(float v) { mLastClusterZ = v; }
  void setSide(Side v) { mSide = v; }
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
  const o2::DataFormat::TPC::ClusterNative& getCluster(int nCluster,
                                                       const o2::DataFormat::TPC::ClusterNativeAccessFullTPC& clusters,
                                                       uint8_t& sectorIndex, uint8_t& rowIndex) const
  {
    uint32_t clusterIndex;
    getClusterReference(nCluster, sectorIndex, rowIndex, clusterIndex);
    return (clusters.clusters[sectorIndex][rowIndex][clusterIndex]);
  }
  const o2::DataFormat::TPC::ClusterNative& getCluster(
    int nCluster, const o2::DataFormat::TPC::ClusterNativeAccessFullTPC& clusters) const
  {
    uint8_t sectorIndex, rowIndex;
    return (getCluster(nCluster, clusters, sectorIndex, rowIndex));
  }

 private:
  std::vector<Cluster> mClusterVector;
  float mTime0 =
    0.f; // Reference Z of the track assumed for the vertex, scaled with pseudo VDrift and reference timeframe length.
  float mLastClusterZ = 0.f;    // Z position of last cluster
  char mSide = Side::UNDEFINED; // TPC Side (A or C) where the track is located (seeding start of the track for those
                                // crossing the central electrode)
  float mChi2 = 0.f;            // Chi2 of the track
  o2::track::TrackParCov mOuterParam; // Track parameters at outer end of TPC.

  // New structure to store cluster references
  int mNClusters = 0;
  std::vector<uint32_t> mClusterReferences;

  ClassDefNV(TrackTPC, 1);
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

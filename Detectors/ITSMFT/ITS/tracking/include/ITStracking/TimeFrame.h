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
///

#ifndef TRACKINGITSU_INCLUDE_TIMEFRAME_H_
#define TRACKINGITSU_INCLUDE_TIMEFRAME_H_

#include <array>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <gsl/gsl>

#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITS/Vertex.h"

#include "ITStracking/Cell.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/ExternalAllocator.h"
#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/ROFLookupTables.h"
#include "ITStracking/TrackingTopology.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace gpu
{
class GPUChainITS;
}

namespace itsmft
{
class Cluster;
class CompClusterExt;
class TopologyDictionary;
class ROFRecord;
} // namespace itsmft

namespace its
{
namespace gpu
{
template <int>
class TimeFrameGPU;
}

template <int NLayers>
struct TimeFrame {
  using IndexTableUtilsN = IndexTableUtils<NLayers>;
  using ROFOverlapTableN = ROFOverlapTable<NLayers>;
  using ROFVertexLookupTableN = ROFVertexLookupTable<NLayers>;
  using ROFMaskTableN = ROFMaskTable<NLayers>;
  using TrackingTopologyN = TrackingTopology<NLayers>;
  using TrackSeedN = TrackSeed<NLayers>;
  friend class gpu::TimeFrameGPU<NLayers>;

  TimeFrame() = default;
  virtual ~TimeFrame() = default;

  const Vertex& getPrimaryVertex(const int ivtx) const { return mPrimaryVertices[ivtx]; }
  auto& getPrimaryVertices() { return mPrimaryVertices; };
  auto getPrimaryVerticesNum() { return mPrimaryVertices.size(); };
  const auto& getPrimaryVertices() const { return mPrimaryVertices; };
  auto& getPrimaryVerticesLabels() { return mPrimaryVerticesLabels; };
  gsl::span<const Vertex> getPrimaryVertices(int layer, int rofId) const;
  void addPrimaryVertex(const Vertex& vertex);
  void addPrimaryVertexLabel(const VertexLabel& label) { mPrimaryVerticesLabels.push_back(label); }

  // read-in data
  void loadROFrameData(gsl::span<const o2::itsmft::ROFRecord> rofs,
                       gsl::span<const itsmft::CompClusterExt> clusters,
                       gsl::span<const unsigned char>::iterator& pattIt,
                       const itsmft::TopologyDictionary* dict,
                       int layer,
                       const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr);
  void resetROFrameData(int iLayer);
  void prepareROFrameData(gsl::span<const itsmft::CompClusterExt> clusters, int layer);

  int getTotalClusters() const;
  bool empty() const { return getTotalClusters() == 0; }
  int getSortedIndex(int rofId, int layer, int idx) const { return mROFramesClusters[layer][rofId] + idx; }
  int getSortedStartIndex(const int rofId, const int layer) const { return mROFramesClusters[layer][rofId]; }
  int getNrof(int layer) const { return mROFramesClusters[layer].size() - 1; }

  void resetBeamXY(const float x, const float y, const float w = 0);
  void setBeamPosition(const float x, const float y, const float s2, const float base = 50.f, const float systematic = 0.f)
  {
    isBeamPositionOverridden = true;
    resetBeamXY(x, y, s2 / o2::gpu::CAMath::Sqrt((base * base) + systematic));
  }

  float getBeamX() const { return mBeamPos[0]; }
  float getBeamY() const { return mBeamPos[1]; }
  std::array<float, 2>& getBeamXY() { return mBeamPos; }

  auto& getMinRs() { return mMinR; }
  auto& getMaxRs() { return mMaxR; }
  float getMinR(int layer) const { return mMinR[layer]; }
  float getMaxR(int layer) const { return mMaxR[layer]; }
  float getLinkPhiCut(int linkId) const { return mLinkPhiCuts[linkId]; }
  float getLinkMSAngle(int linkId) const { return mLinkMSAngles[linkId]; }
  auto& getLinkPhiCuts() { return mLinkPhiCuts; }
  auto& getLinkMSAngles() { return mLinkMSAngles; }
  float getPositionResolution(int layer) const { return mPositionResolution[layer]; }
  auto& getPositionResolutions() { return mPositionResolution; }

  gsl::span<Cluster> getClustersOnLayer(int rofId, int layerId);
  gsl::span<const Cluster> getClustersOnLayer(int rofId, int layerId) const;
  gsl::span<const Cluster> getClustersPerROFrange(int rofMin, int range, int layerId) const;
  gsl::span<const Cluster> getUnsortedClustersOnLayer(int rofId, int layerId) const;
  gsl::span<uint8_t> getUsedClustersROF(int rofId, int layerId);
  gsl::span<const uint8_t> getUsedClustersROF(int rofId, int layerId) const;
  gsl::span<const int> getROFramesClustersPerROFrange(int rofMin, int range, int layerId) const;
  gsl::span<const int> getROFrameClusters(int layerId) const;
  gsl::span<const int> getNClustersROFrange(int rofMin, int range, int layerId) const;
  gsl::span<int> getIndexTable(int rofId, int layerId);
  const auto& getTrackingFrameInfoOnLayer(int layerId) const { return mTrackingFrameInfo[layerId]; }

  // navigation tables
  const auto& getIndexTableUtils() const { return mIndexTableUtils; }
  const auto& getROFOverlapTable() const { return mROFOverlapTable; }
  const auto& getROFOverlapTableView() const { return mROFOverlapTableView; }
  const auto& getTrackerTopologies() const { return mTrackerTopologies; }
  const auto& getTrackingTopologyView() const { return mTrackingTopologyView; }
  void setROFOverlapTable(ROFOverlapTableN table)
  {
    mROFOverlapTable = std::move(table);
    mROFOverlapTableView = mROFOverlapTable.getView();
  }
  const auto& getROFVertexLookupTable() const { return mROFVertexLookupTable; }
  const auto& getROFVertexLookupTableView() const { return mROFVertexLookupTableView; }
  void setROFVertexLookupTable(ROFVertexLookupTableN table)
  {
    mROFVertexLookupTable = std::move(table);
    mROFVertexLookupTableView = mROFVertexLookupTable.getView();
  }
  void updateROFVertexLookupTable() { mROFVertexLookupTable.update(mPrimaryVertices.data(), mPrimaryVertices.size()); }
  void setMultiplicityCutMask(ROFMaskTableN cutMask)
  {
    mMultiplicityCutMask = std::move(cutMask);
    mROFMaskView = mROFMask->getView();
  }
  void useMultiplictyMask() noexcept
  {
    mROFMask = &mMultiplicityCutMask;
    mROFMaskView = mROFMask->getView();
  }
  void setUPCCutMask(ROFMaskTableN cutMask) { mUPCCutMask = std::move(cutMask); }
  void useUPCMask() noexcept
  {
    mROFMask = &mUPCCutMask;
    mROFMaskView = mROFMask->getView();
  }
  const auto& getROFMaskView() const { return mROFMaskView; }

  const TrackingFrameInfo& getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const;
  gsl::span<const MCCompLabel> getClusterLabels(int layerId, const Cluster& cl) const { return getClusterLabels(layerId, cl.clusterId); }
  gsl::span<const MCCompLabel> getClusterLabels(int layerId, const int clId) const { return mClusterLabels[((mIsStaggered) ? layerId : 0)]->getLabels(mClusterExternalIndices[layerId][clId]); }
  int getClusterExternalIndex(int layerId, const int clId) const { return mClusterExternalIndices[layerId][clId]; }
  int getClusterSize(int layer, int clusterId) const { return mClusterSize[layer][clusterId]; }
  void setClusterSize(int layer, bounded_vector<uint8_t>& v) { mClusterSize[layer] = std::move(v); }

  auto& getTrackletsLabel(int layer) { return mTrackletLabels[layer]; }
  auto& getCellsLabel(int layer) { return mCellLabels[layer]; }

  bool hasMCinformation() const { return mClusterLabels[0] != nullptr; }
  void initVertexingTopology(const TrackingParameters& trkParam);
  void initDefaultTrackingTopology(const TrackingParameters& trkParam, const int maxLayers = NLayers);
  void initTrackerTopologies(gsl::span<const TrackingParameters> trkParams, const int maxLayers = NLayers);
  void initialise(const TrackingParameters& trkParam, const int maxLayers = NLayers, const int iteration = constants::UnusedIndex);

  bool isClusterUsed(int layer, int clusterId) const { return mUsedClusters[layer][clusterId]; }
  void markUsedCluster(int layer, int clusterId) { mUsedClusters[layer][clusterId] = true; }
  gsl::span<unsigned char> getUsedClusters(const int layer);

  auto& getTracklets() { return mTracklets; }
  auto& getTrackletsLookupTable() { return mTrackletsLookupTable; }

  auto& getClusters() { return mClusters; }
  auto& getUnsortedClusters() { return mUnsortedClusters; }
  int getClusterROF(int iLayer, int iCluster);
  auto& getCells() { return mCells; }

  auto& getCellsLookupTable() { return mCellsLookupTable; }
  auto& getCellsNeighbours() { return mCellsNeighbours; }
  auto& getCellsNeighboursTopology() { return mCellsNeighboursTopology; }
  auto& getCellsNeighboursLUT() { return mCellsNeighboursLUT; }
  auto& getTracks() { return mTracks; }
  auto& getTracksLabel() { return mTracksLabel; }
  auto& getLinesLabel(const int rofId) { return mLinesLabels[rofId]; }

  size_t getNumberOfClusters() const;
  virtual size_t getNumberOfCells() const;
  virtual size_t getNumberOfTracklets() const;
  virtual size_t getNumberOfNeighbours() const;
  size_t getNumberOfTracks() const;
  size_t getNumberOfUsedClusters() const;
  void resetTrackExtensionCounters();
  void addTrackExtensionCounters(size_t nTracks, size_t nClusters);
  size_t getNExtendedTracks() const { return mNExtendedTracks; }
  size_t getNExtendedClusters() const { return mNExtendedClusters; }

  /// memory management
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool);
  auto& getMemoryPool() const noexcept { return mMemoryPool; }
  bool checkMemory(unsigned long max) { return getArtefactsMemory() < max; }
  unsigned long getArtefactsMemory() const;
  void printArtefactsMemory() const;

  /// staggering
  void setIsStaggered(bool b) noexcept { mIsStaggered = b; }

  // Vertexer
  void computeTrackletsPerROFScans();
  void computeTracletsPerClusterScans();
  int& getNTrackletsROF(int rofId, int combId) { return mNTrackletsPerROF[combId][rofId]; }
  auto& getLines(int rofId) { return mLines[rofId]; }
  int getNLinesTotal() const noexcept { return mTotalLines; }
  void setNLinesTotal(uint32_t a) noexcept { mTotalLines = a; }
  auto& getTrackletClusters(int rofId) { return mTrackletClusters[rofId]; }
  gsl::span<const Tracklet> getFoundTracklets(int rofId, int combId) const;
  gsl::span<Tracklet> getFoundTracklets(int rofId, int combId);
  gsl::span<const MCCompLabel> getLabelsFoundTracklets(int rofId, int combId) const;
  gsl::span<int> getNTrackletsCluster(int rofId, int combId);
  gsl::span<int> getExclusiveNTrackletsCluster(int rofId, int combId);
  uint32_t getTotalTrackletsTF(const int iLayer) { return mTotalTracklets[iLayer]; }
  int getTotalClustersPerROFrange(int rofMin, int range, int layerId) const;
  // \Vertexer

  int hasBogusClusters() const { return std::accumulate(mBogusClusters.begin(), mBogusClusters.end(), 0); }

  void setBz(float bz) { mBz = bz; }
  float getBz() const { return mBz; }

  /// State if memory will be externally managed by the GPU framework
  ExternalAllocator* mExternalAllocator{nullptr};
  std::shared_ptr<BoundedMemoryResource> mExtMemoryPool; // host memory pool managed by the framework
  auto getFrameworkAllocator() { return mExternalAllocator; };
  void setFrameworkAllocator(ExternalAllocator* ext);
  bool hasFrameworkAllocator() const noexcept { return mExternalAllocator != nullptr; }
  std::pmr::memory_resource* getMaybeFrameworkHostResource(bool forceHost = false) { return (hasFrameworkAllocator() && !forceHost) ? mExtMemoryPool.get() : mMemoryPool.get(); }

  // Propagator
  const o2::base::PropagatorImpl<float>* getDevicePropagator() const { return mPropagatorDevice; }
  virtual void setDevicePropagator(const o2::base::PropagatorImpl<float>* /*unused*/) {};

  template <typename... T>
  void addClusterToLayer(int layer, T&&... args);
  template <typename... T>
  void addTrackingFrameInfoToLayer(int layer, T&&... args);
  void addClusterExternalIndexToLayer(int layer, const int idx) { mClusterExternalIndices[layer].push_back(idx); }

  std::array<bounded_vector<Cluster>, NLayers> mClusters;
  std::array<bounded_vector<TrackingFrameInfo>, NLayers> mTrackingFrameInfo;
  std::array<bounded_vector<int>, NLayers> mClusterExternalIndices;
  std::array<bounded_vector<int>, NLayers> mROFramesClusters;
  std::array<const dataformats::MCTruthContainer<MCCompLabel>*, NLayers> mClusterLabels{nullptr};
  std::array<bounded_vector<int>, 2> mNTrackletsPerCluster;
  std::array<bounded_vector<int>, 2> mNTrackletsPerClusterSum;
  std::array<bounded_vector<int>, NLayers> mNClustersPerROF;
  std::array<bounded_vector<int>, NLayers> mIndexTables;
  std::vector<bounded_vector<int>> mTrackletsLookupTable;
  std::array<bounded_vector<uint8_t>, NLayers> mUsedClusters;

  std::array<bounded_vector<Cluster>, NLayers> mUnsortedClusters;
  std::vector<bounded_vector<Tracklet>> mTracklets;
  std::vector<bounded_vector<CellSeed>> mCells;
  bounded_vector<TrackITSExt> mTracks;
  bounded_vector<MCCompLabel> mTracksLabel;
  size_t mNExtendedTracks = 0;
  size_t mNExtendedClusters = 0;
  std::vector<bounded_vector<int>> mCellsNeighbours;
  std::vector<bounded_vector<int>> mCellsNeighboursTopology;
  std::vector<bounded_vector<int>> mCellsLookupTable;

  const o2::base::PropagatorImpl<float>* mPropagatorDevice = nullptr; // Needed only for GPU

  virtual void wipe();

  // interface
  virtual bool isGPU() const noexcept { return false; }
  virtual const char* getName() const noexcept { return "CPU"; }

 protected:
  void prepareClusters(const TrackingParameters& trkParam, const int maxLayers = NLayers);
  float mBz = 5.;
  unsigned int mNTotalLowPtVertices = 0;
  int mBeamPosWeight = 0;
  std::array<float, 2> mBeamPos = {0.f, 0.f};
  bool isBeamPositionOverridden = false;
  std::array<float, NLayers> mMinR;
  std::array<float, NLayers> mMaxR;
  bounded_vector<float> mLinkPhiCuts;
  bounded_vector<float> mLinkMSAngles;
  bounded_vector<float> mPositionResolution;
  std::array<bounded_vector<uint8_t>, NLayers> mClusterSize;

  bounded_vector<std::array<float, 2>> mPValphaX; /// PV x and alpha for track propagation
  std::vector<bounded_vector<MCCompLabel>> mTrackletLabels;
  std::vector<bounded_vector<MCCompLabel>> mCellLabels;
  std::vector<bounded_vector<int>> mCellsNeighboursLUT;
  bounded_vector<int> mBogusClusters; /// keep track of clusters with wild coordinates

  // Vertexer
  bounded_vector<Vertex> mPrimaryVertices;
  bounded_vector<VertexLabel> mPrimaryVerticesLabels;
  std::vector<bounded_vector<int>> mNTrackletsPerROF;
  std::vector<bounded_vector<Line>> mLines;
  std::vector<bounded_vector<ClusterLines>> mTrackletClusters;
  std::array<bounded_vector<int>, 2> mTrackletsIndexROF;
  std::vector<bounded_vector<MCCompLabel>> mLinesLabels;
  std::array<uint32_t, 2> mTotalTracklets = {0, 0};
  uint32_t mTotalLines = 0;
  // \Vertexer

  // lookup tables
  IndexTableUtilsN mIndexTableUtils;
  ROFOverlapTableN mROFOverlapTable;
  ROFOverlapTableN::View mROFOverlapTableView;
  TrackingTopologyN mVertexingTopology;
  TrackingTopologyN mDefaultTrackingTopology;
  std::vector<TrackingTopologyN> mTrackerTopologies;
  typename TrackingTopologyN::View mTrackingTopologyView;
  ROFVertexLookupTableN mROFVertexLookupTable;
  ROFVertexLookupTableN::View mROFVertexLookupTableView;
  ROFMaskTableN mMultiplicityCutMask;
  ROFMaskTableN mUPCCutMask;
  ROFMaskTableN* mROFMask = &mMultiplicityCutMask;
  ROFMaskTableN::View mROFMaskView;

  bool mIsStaggered{false};

  std::shared_ptr<BoundedMemoryResource> mMemoryPool;
};

template <int NLayers>
gsl::span<const Vertex> TimeFrame<NLayers>::getPrimaryVertices(int layer, int rofId) const
{
  if (rofId < 0 || rofId >= getNrof(layer)) {
    return {};
  }
  const auto& entry = mROFVertexLookupTableView.getVertices(layer, rofId);
  return {&mPrimaryVertices[entry.getFirstEntry()], static_cast<gsl::span<const Vertex>::size_type>(entry.getEntries())};
}

template <int NLayers>
inline void TimeFrame<NLayers>::resetBeamXY(const float x, const float y, const float w)
{
  mBeamPos[0] = x;
  mBeamPos[1] = y;
  mBeamPosWeight = w;
}

template <int NLayers>
inline gsl::span<const int> TimeFrame<NLayers>::getROFrameClusters(int layerId) const
{
  return {&mROFramesClusters[layerId][0], static_cast<gsl::span<const int>::size_type>(mROFramesClusters[layerId].size())};
}

template <int NLayers>
inline gsl::span<Cluster> TimeFrame<NLayers>::getClustersOnLayer(int rofId, int layerId)
{
  if (rofId < 0 || rofId >= getNrof(layerId)) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mClusters[layerId][startIdx], static_cast<gsl::span<Cluster>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<const Cluster> TimeFrame<NLayers>::getClustersOnLayer(int rofId, int layerId) const
{
  if (rofId < 0 || rofId >= getNrof(layerId)) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mClusters[layerId][startIdx], static_cast<gsl::span<const Cluster>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<uint8_t> TimeFrame<NLayers>::getUsedClustersROF(int rofId, int layerId)
{
  if (rofId < 0 || rofId >= getNrof(layerId)) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mUsedClusters[layerId][startIdx], static_cast<gsl::span<uint8_t>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<const uint8_t> TimeFrame<NLayers>::getUsedClustersROF(int rofId, int layerId) const
{
  if (rofId < 0 || rofId >= getNrof(layerId)) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mUsedClusters[layerId][startIdx], static_cast<gsl::span<const uint8_t>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<const Cluster> TimeFrame<NLayers>::getClustersPerROFrange(int rofMin, int range, int layerId) const
{
  if (rofMin < 0 || rofMin >= getNrof(layerId)) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofMin]}; // First cluster of rofMin
  int endIdx{mROFramesClusters[layerId][o2::gpu::CAMath::Min(rofMin + range, getNrof(layerId))]};
  return {&mClusters[layerId][startIdx], static_cast<gsl::span<Cluster>::size_type>(endIdx - startIdx)};
}

template <int NLayers>
inline gsl::span<const int> TimeFrame<NLayers>::getROFramesClustersPerROFrange(int rofMin, int range, int layerId) const
{
  int chkdRange{o2::gpu::CAMath::Min(range, getNrof(layerId) - rofMin)};
  return {&mROFramesClusters[layerId][rofMin], static_cast<gsl::span<int>::size_type>(chkdRange)};
}

template <int NLayers>
inline gsl::span<const int> TimeFrame<NLayers>::getNClustersROFrange(int rofMin, int range, int layerId) const
{
  int chkdRange{o2::gpu::CAMath::Min(range, getNrof(layerId) - rofMin)};
  return {&mNClustersPerROF[layerId][rofMin], static_cast<gsl::span<int>::size_type>(chkdRange)};
}

template <int NLayers>
inline int TimeFrame<NLayers>::getTotalClustersPerROFrange(int rofMin, int range, int layerId) const
{
  int startIdx{rofMin}; // First cluster of rofMin
  int endIdx{o2::gpu::CAMath::Min(rofMin + range, getNrof(layerId))};
  return mROFramesClusters[layerId][endIdx] - mROFramesClusters[layerId][startIdx];
}

template <int NLayers>
inline int TimeFrame<NLayers>::getClusterROF(int iLayer, int iCluster)
{
  return std::lower_bound(mROFramesClusters[iLayer].begin(), mROFramesClusters[iLayer].end(), iCluster + 1) - mROFramesClusters[iLayer].begin() - 1;
}

template <int NLayers>
inline gsl::span<const Cluster> TimeFrame<NLayers>::getUnsortedClustersOnLayer(int rofId, int layerId) const
{
  if (rofId < 0 || rofId >= getNrof(layerId)) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mUnsortedClusters[layerId][startIdx], static_cast<gsl::span<Cluster>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<int> TimeFrame<NLayers>::getIndexTable(int rofId, int layer)
{
  if (rofId < 0 || rofId >= getNrof(layer)) {
    return {};
  }
  const int tableSize = mIndexTableUtils.getNphiBins() * mIndexTableUtils.getNzBins() + 1;
  return {&mIndexTables[layer][rofId * tableSize], static_cast<gsl::span<int>::size_type>(tableSize)};
}

template <int NLayers>
template <typename... T>
void TimeFrame<NLayers>::addClusterToLayer(int layer, T&&... values)
{
  mUnsortedClusters[layer].emplace_back(std::forward<T>(values)...);
}

template <int NLayers>
template <typename... T>
void TimeFrame<NLayers>::addTrackingFrameInfoToLayer(int layer, T&&... values)
{
  mTrackingFrameInfo[layer].emplace_back(std::forward<T>(values)...);
}

template <int NLayers>
inline gsl::span<uint8_t> TimeFrame<NLayers>::getUsedClusters(const int layer)
{
  return {&mUsedClusters[layer][0], static_cast<gsl::span<uint8_t>::size_type>(mUsedClusters[layer].size())};
}

template <int NLayers>
inline gsl::span<int> TimeFrame<NLayers>::getNTrackletsCluster(int rofId, int combId)
{
  if (rofId < 0 || rofId >= getNrof(1)) {
    return {};
  }
  auto startIdx{mROFramesClusters[1][rofId]};
  return {&mNTrackletsPerCluster[combId][startIdx], static_cast<gsl::span<int>::size_type>(mROFramesClusters[1][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<int> TimeFrame<NLayers>::getExclusiveNTrackletsCluster(int rofId, int combId)
{
  if (rofId < 0 || rofId >= getNrof(1)) {
    return {};
  }
  auto clusStartIdx{mROFramesClusters[1][rofId]};

  return {&mNTrackletsPerClusterSum[combId][clusStartIdx], static_cast<gsl::span<int>::size_type>(mROFramesClusters[1][rofId + 1] - clusStartIdx)};
}

template <int NLayers>
inline gsl::span<Tracklet> TimeFrame<NLayers>::getFoundTracklets(int rofId, int combId)
{
  if (rofId < 0 || rofId >= getNrof(1) || mTracklets[combId].empty()) {
    return {};
  }
  auto startIdx{mNTrackletsPerROF[combId][rofId]};
  return {&mTracklets[combId][startIdx], static_cast<gsl::span<Tracklet>::size_type>(mNTrackletsPerROF[combId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<const Tracklet> TimeFrame<NLayers>::getFoundTracklets(int rofId, int combId) const
{
  if (rofId < 0 || rofId >= getNrof(1)) {
    return {};
  }
  auto startIdx{mNTrackletsPerROF[combId][rofId]};
  return {&mTracklets[combId][startIdx], static_cast<gsl::span<Tracklet>::size_type>(mNTrackletsPerROF[combId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline gsl::span<const MCCompLabel> TimeFrame<NLayers>::getLabelsFoundTracklets(int rofId, int combId) const
{
  if (rofId < 0 || rofId >= getNrof(1) || !hasMCinformation()) {
    return {};
  }
  auto startIdx{mNTrackletsPerROF[combId][rofId]};
  return {&mTrackletLabels[combId][startIdx], static_cast<gsl::span<Tracklet>::size_type>(mNTrackletsPerROF[combId][rofId + 1] - startIdx)};
}

template <int NLayers>
inline int TimeFrame<NLayers>::getTotalClusters() const
{
  size_t totalClusters{0};
  for (const auto& clusters : mUnsortedClusters) {
    totalClusters += clusters.size();
  }
  return int(totalClusters);
}

template <int NLayers>
inline size_t TimeFrame<NLayers>::getNumberOfClusters() const
{
  size_t nClusters{0};
  for (const auto& layer : mClusters) {
    nClusters += layer.size();
  }
  return nClusters;
}

template <int NLayers>
inline size_t TimeFrame<NLayers>::getNumberOfCells() const
{
  size_t nCells{0};
  for (const auto& layer : mCells) {
    nCells += layer.size();
  }
  return nCells;
}

template <int NLayers>
inline size_t TimeFrame<NLayers>::getNumberOfTracklets() const
{
  size_t nTracklets{0};
  for (const auto& layer : mTracklets) {
    nTracklets += layer.size();
  }
  return nTracklets;
}

template <int NLayers>
inline size_t TimeFrame<NLayers>::getNumberOfNeighbours() const
{
  size_t neigh{0};
  for (const auto& l : mCellsNeighbours) {
    neigh += l.size();
  }
  return neigh;
}

template <int NLayers>
inline size_t TimeFrame<NLayers>::getNumberOfTracks() const
{
  return mTracks.size();
}

template <int NLayers>
inline size_t TimeFrame<NLayers>::getNumberOfUsedClusters() const
{
  size_t nClusters = 0;
  for (const auto& layer : mUsedClusters) {
    nClusters += std::count(layer.begin(), layer.end(), true);
  }
  return nClusters;
}

template <int NLayers>
inline void TimeFrame<NLayers>::resetTrackExtensionCounters()
{
  mNExtendedTracks = 0;
  mNExtendedClusters = 0;
}

template <int NLayers>
inline void TimeFrame<NLayers>::addTrackExtensionCounters(size_t nTracks, size_t nClusters)
{
  mNExtendedTracks += nTracks;
  mNExtendedClusters += nClusters;
}

} // namespace its
} // namespace o2

#endif

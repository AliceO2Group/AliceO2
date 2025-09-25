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

#include "ITStracking/Cell.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Constants.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Road.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/ExternalAllocator.h"
#include "ITStracking/BoundedAllocator.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "ReconstructionDataFormats/Vertex.h"
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

template <int nLayers = 7>
struct TimeFrame {
  using IndexTableUtilsN = IndexTableUtils<nLayers>;
  using CellSeedN = CellSeed<nLayers>;
  friend class gpu::TimeFrameGPU<nLayers>;

  TimeFrame() = default;
  virtual ~TimeFrame() = default;

  const Vertex& getPrimaryVertex(const int ivtx) const { return mPrimaryVertices[ivtx]; }
  gsl::span<const Vertex> getPrimaryVertices(int rofId) const;
  gsl::span<const Vertex> getPrimaryVertices(int romin, int romax) const;
  gsl::span<const std::pair<MCCompLabel, float>> getPrimaryVerticesMCRecInfo(const int rofId) const;
  gsl::span<const MCCompLabel> getPrimaryVerticesContributors(const int rofId) const;
  gsl::span<const std::array<float, 2>> getPrimaryVerticesXAlpha(int rofId) const;
  void fillPrimaryVerticesXandAlpha();
  int getPrimaryVerticesNum(int rofId = -1) const;
  void addPrimaryVerticesLabels(bounded_vector<std::pair<MCCompLabel, float>>& labels);
  void addPrimaryVerticesContributorLabels(bounded_vector<MCCompLabel>& labels);
  void addPrimaryVertices(const bounded_vector<Vertex>& vertices, const int iteration);
  void addPrimaryVerticesInROF(const bounded_vector<Vertex>& vertices, const int rofId, const int iteration);
  void addPrimaryVerticesLabelsInROF(const bounded_vector<std::pair<MCCompLabel, float>>& labels, const int rofId);
  void addPrimaryVerticesContributorLabelsInROF(const bounded_vector<MCCompLabel>& labels, const int rofId);
  void removePrimaryVerticesInROf(const int rofId);
  int loadROFrameData(const o2::itsmft::ROFRecord& rof, gsl::span<const itsmft::Cluster> clusters,
                      const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr);

  int loadROFrameData(gsl::span<o2::itsmft::ROFRecord> rofs,
                      gsl::span<const itsmft::CompClusterExt> clusters,
                      gsl::span<const unsigned char>::iterator& pattIt,
                      const itsmft::TopologyDictionary* dict,
                      const dataformats::MCTruthContainer<MCCompLabel>* mcLabels = nullptr);
  void resetROFrameData(size_t nROFs);

  int getTotalClusters() const;
  auto& getTotVertIteration() { return mTotVertPerIteration; }
  bool empty() const { return getTotalClusters() == 0; }
  int getSortedIndex(int rofId, int layer, int idx) const { return mROFramesClusters[layer][rofId] + idx; }
  int getSortedStartIndex(const int rofId, const int layer) const { return mROFramesClusters[layer][rofId]; }
  int getNrof() const { return mNrof; }

  void resetBeamXY(const float x, const float y, const float w = 0);
  void setBeamPosition(const float x, const float y, const float s2, const float base = 50.f, const float systematic = 0.f)
  {
    isBeamPositionOverridden = true;
    resetBeamXY(x, y, s2 / o2::gpu::CAMath::Sqrt(base * base + systematic));
  }

  float getBeamX() const { return mBeamPos[0]; }
  float getBeamY() const { return mBeamPos[1]; }
  auto& getMinRs() { return mMinR; }
  auto& getMaxRs() { return mMaxR; }
  float getMinR(int layer) const { return mMinR[layer]; }
  float getMaxR(int layer) const { return mMaxR[layer]; }
  float getMSangle(int layer) const { return mMSangles[layer]; }
  auto& getMSangles() { return mMSangles; }
  float getPhiCut(int layer) const { return mPhiCuts[layer]; }
  auto& getPhiCuts() { return mPhiCuts; }
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
  gsl::span<const int> getIndexTablePerROFrange(int rofMin, int range, int layerId) const;
  gsl::span<int> getIndexTable(int rofId, int layerId);
  auto& getIndexTableWhole(int layerId) { return mIndexTables[layerId]; }
  const auto& getTrackingFrameInfoOnLayer(int layerId) const { return mTrackingFrameInfo[layerId]; }

  const TrackingFrameInfo& getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const;
  gsl::span<const MCCompLabel> getClusterLabels(int layerId, const Cluster& cl) const { return getClusterLabels(layerId, cl.clusterId); }
  gsl::span<const MCCompLabel> getClusterLabels(int layerId, const int clId) const { return mClusterLabels->getLabels(mClusterExternalIndices[layerId][clId]); }
  int getClusterExternalIndex(int layerId, const int clId) const { return mClusterExternalIndices[layerId][clId]; }
  int getClusterSize(int clusterId) const { return mClusterSize[clusterId]; }
  void setClusterSize(bounded_vector<uint8_t>& v) { mClusterSize = std::move(v); }

  auto& getTrackletsLabel(int layer) { return mTrackletLabels[layer]; }
  auto& getCellsLabel(int layer) { return mCellLabels[layer]; }

  bool hasMCinformation() const { return mClusterLabels; }
  void initialise(const int iteration, const TrackingParameters& trkParam, const int maxLayers = 7, bool resetVertices = true);
  void resetRofPV()
  {
    deepVectorClear(mPrimaryVertices);
    mROFramesPV.resize(1, 0);
    mTotVertPerIteration.resize(1);
  }

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
  auto& getCellsNeighboursLUT() { return mCellsNeighboursLUT; }
  auto& getRoads() { return mRoads; }
  auto& getTracks(int rofId) { return mTracks[rofId]; }
  auto& getTracksLabel(const int rofId) { return mTracksLabel[rofId]; }
  auto& getLinesLabel(const int rofId) { return mLinesLabels[rofId]; }
  auto& getVerticesMCRecInfo() { return mVerticesMCRecInfo; }

  int getNumberOfClusters() const;
  virtual int getNumberOfCells() const;
  virtual int getNumberOfTracklets() const;
  virtual int getNumberOfNeighbours() const;
  size_t getNumberOfTracks() const;
  size_t getNumberOfUsedClusters() const;
  auto getNumberOfExtendedTracks() const { return mNExtendedTracks; }
  auto getNumberOfUsedExtendedClusters() const { return mNExtendedUsedClusters; }

  /// memory management
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool);
  auto& getMemoryPool() const noexcept { return mMemoryPool; }
  bool checkMemory(unsigned long max) { return getArtefactsMemory() < max; }
  unsigned long getArtefactsMemory() const;
  void printArtefactsMemory() const;

  /// ROF cuts
  int getROFCutClusterMult() const { return mCutClusterMult; };
  int getROFCutVertexMult() const { return mCutVertexMult; };
  int getROFCutAllMult() const { return mCutClusterMult + mCutVertexMult; }

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
  std::array<float, 2>& getBeamXY() { return mBeamPos; }
  unsigned int& getNoVertexROF() { return mNoVertexROF; }
  void insertPastVertex(const Vertex& vertex, const int refROFId);
  // \Vertexer

  void initialiseRoadLabels();
  void setRoadLabel(int i, const unsigned long long& lab, bool fake);
  const unsigned long long& getRoadLabel(int i) const { return mRoadLabels[i].first; }
  bool isRoadFake(int i) const { return mRoadLabels[i].second; }

  void setMultiplicityCutMask(const std::vector<uint8_t>& cutMask) { mMultiplicityCutMask = cutMask; }
  void setROFMask(const std::vector<uint8_t>& rofMask) { mROFMask = rofMask; }
  void swapMasks() { mMultiplicityCutMask.swap(mROFMask); }

  int hasBogusClusters() const { return std::accumulate(mBogusClusters.begin(), mBogusClusters.end(), 0); }

  void setBz(float bz) { mBz = bz; }
  float getBz() const { return mBz; }

  /// State if memory will be externally managed.
  // device
  ExternalAllocator* mExtDeviceAllocator{nullptr};
  void setExternalDeviceAllocator(ExternalAllocator* allocator) { mExtDeviceAllocator = allocator; }
  ExternalAllocator* getExternalDeviceAllocator() { return mExtDeviceAllocator; }
  bool hasExternalDeviceAllocator() const noexcept { return mExtDeviceAllocator != nullptr; }
  // host
  ExternalAllocator* mExtHostAllocator{nullptr};
  void setExternalHostAllocator(ExternalAllocator* allocator)
  {
    mExtHostAllocator = allocator;
    mExtMemoryPool = std::make_shared<BoundedMemoryResource>(mExtHostAllocator);
  }
  ExternalAllocator* getExternalHostAllocator() { return mExtHostAllocator; }
  bool hasExternalHostAllocator() const noexcept { return mExtHostAllocator != nullptr; }
  std::shared_ptr<BoundedMemoryResource> mExtMemoryPool;
  std::pmr::memory_resource* getMaybeExternalHostResource(bool forceHost = false) { return (hasExternalHostAllocator() && !forceHost) ? mExtMemoryPool.get() : mMemoryPool.get(); }
  // Propagator
  const o2::base::PropagatorImpl<float>* getDevicePropagator() const { return mPropagatorDevice; }
  virtual void setDevicePropagator(const o2::base::PropagatorImpl<float>*) {};

  template <typename... T>
  void addClusterToLayer(int layer, T&&... args);
  template <typename... T>
  void addTrackingFrameInfoToLayer(int layer, T&&... args);
  void addClusterExternalIndexToLayer(int layer, const int idx) { mClusterExternalIndices[layer].push_back(idx); }

  /// Debug and printing
  void checkTrackletLUTs();
  void printROFoffsets();
  void printNClsPerROF();
  void printVertices();
  void printTrackletLUTonLayer(int i);
  void printCellLUTonLayer(int i);
  void printTrackletLUTs();
  void printCellLUTs();
  void printSliceInfo(const int, const int);

  IndexTableUtilsN mIndexTableUtils;

  std::array<bounded_vector<Cluster>, nLayers> mClusters;
  std::array<bounded_vector<TrackingFrameInfo>, nLayers> mTrackingFrameInfo;
  std::array<bounded_vector<int>, nLayers> mClusterExternalIndices;
  std::array<bounded_vector<int>, nLayers> mROFramesClusters;
  const dataformats::MCTruthContainer<MCCompLabel>* mClusterLabels = nullptr;
  std::array<bounded_vector<int>, 2> mNTrackletsPerCluster;
  std::array<bounded_vector<int>, 2> mNTrackletsPerClusterSum;
  std::array<bounded_vector<int>, nLayers> mNClustersPerROF;
  std::array<bounded_vector<int>, nLayers> mIndexTables;
  std::vector<bounded_vector<int>> mTrackletsLookupTable;
  std::array<bounded_vector<uint8_t>, nLayers> mUsedClusters;
  int mNrof = 0;
  int mNExtendedTracks{0};
  int mNExtendedUsedClusters{0};
  bounded_vector<int> mROFramesPV;
  bounded_vector<Vertex> mPrimaryVertices;

  std::array<bounded_vector<Cluster>, nLayers> mUnsortedClusters;
  std::vector<bounded_vector<Tracklet>> mTracklets;
  std::vector<bounded_vector<CellSeedN>> mCells;
  bounded_vector<Road<nLayers - 2>> mRoads;
  std::vector<bounded_vector<TrackITSExt>> mTracks;
  std::vector<bounded_vector<int>> mCellsNeighbours;
  std::vector<bounded_vector<int>> mCellsLookupTable;
  std::vector<uint8_t> mMultiplicityCutMask;

  const o2::base::PropagatorImpl<float>* mPropagatorDevice = nullptr; // Needed only for GPU

  virtual void wipe();

  // interface
  virtual bool isGPU() const noexcept { return false; }
  virtual const char* getName() const noexcept { return "CPU"; }

 private:
  void prepareClusters(const TrackingParameters& trkParam, const int maxLayers = nLayers);
  float mBz = 5.;
  unsigned int mNTotalLowPtVertices = 0;
  int mBeamPosWeight = 0;
  std::array<float, 2> mBeamPos = {0.f, 0.f};
  bool isBeamPositionOverridden = false;
  std::array<float, nLayers> mMinR;
  std::array<float, nLayers> mMaxR;
  bounded_vector<float> mMSangles;
  bounded_vector<float> mPhiCuts;
  bounded_vector<float> mPositionResolution;
  bounded_vector<uint8_t> mClusterSize;

  std::vector<uint8_t> mROFMask;
  bounded_vector<std::array<float, 2>> mPValphaX; /// PV x and alpha for track propagation
  std::vector<bounded_vector<MCCompLabel>> mTrackletLabels;
  std::vector<bounded_vector<MCCompLabel>> mCellLabels;
  std::vector<bounded_vector<int>> mCellsNeighboursLUT;
  std::vector<bounded_vector<MCCompLabel>> mTracksLabel;
  bounded_vector<int> mBogusClusters; /// keep track of clusters with wild coordinates

  bounded_vector<std::pair<unsigned long long, bool>> mRoadLabels;
  int mCutClusterMult{-999};
  int mCutVertexMult{-999};

  // Vertexer
  std::vector<bounded_vector<int>> mNTrackletsPerROF;
  std::vector<bounded_vector<Line>> mLines;
  std::vector<bounded_vector<ClusterLines>> mTrackletClusters;
  std::array<bounded_vector<int>, 2> mTrackletsIndexROF;
  std::vector<bounded_vector<MCCompLabel>> mLinesLabels;
  std::vector<std::pair<MCCompLabel, float>> mVerticesMCRecInfo;
  bounded_vector<MCCompLabel> mVerticesContributorLabels;
  std::array<uint32_t, 2> mTotalTracklets = {0, 0};
  uint32_t mTotalLines = 0;
  unsigned int mNoVertexROF = 0;
  bounded_vector<int> mTotVertPerIteration;
  // \Vertexer

  std::shared_ptr<BoundedMemoryResource> mMemoryPool;
};

template <int nLayers>
inline gsl::span<const Vertex> TimeFrame<nLayers>::getPrimaryVertices(int rofId) const
{
  if (mPrimaryVertices.empty()) {
    return {};
  }
  const int start = mROFramesPV[rofId];
  const int stop_idx = rofId >= mNrof - 1 ? mNrof : rofId + 1;
  int delta = mMultiplicityCutMask[rofId] ? mROFramesPV[stop_idx] - start : 0; // return empty span if Rof is excluded
  return {&mPrimaryVertices[start], static_cast<gsl::span<const Vertex>::size_type>(delta)};
}

template <int nLayers>
inline gsl::span<const std::pair<MCCompLabel, float>> TimeFrame<nLayers>::getPrimaryVerticesMCRecInfo(const int rofId) const
{
  const int start = mROFramesPV[rofId];
  const int stop_idx = rofId >= mNrof - 1 ? mNrof : rofId + 1;
  int delta = mMultiplicityCutMask[rofId] ? mROFramesPV[stop_idx] - start : 0; // return empty span if Rof is excluded
  return {&(mVerticesMCRecInfo[start]), static_cast<gsl::span<const std::pair<MCCompLabel, float>>::size_type>(delta)};
}

template <int nLayers>
inline gsl::span<const MCCompLabel> TimeFrame<nLayers>::getPrimaryVerticesContributors(const int rofId) const
{
  // count the number of cont. in rofs before target rof
  unsigned int start{0}, delta{0};
  const auto& pvsBefore = getPrimaryVertices(0, rofId - 1);
  for (const auto& pv : pvsBefore) {
    start += pv.getNContributors();
  }
  const auto& pvsIn = getPrimaryVertices(rofId);
  for (const auto& pv : pvsIn) {
    delta += pv.getNContributors();
  }
  return {&(mVerticesContributorLabels[start]), static_cast<gsl::span<const MCCompLabel>::size_type>(delta)};
}

template <int nLayers>
inline gsl::span<const Vertex> TimeFrame<nLayers>::getPrimaryVertices(int romin, int romax) const
{
  if (mPrimaryVertices.empty()) {
    return {};
  }
  const int stop_idx = romax >= mNrof - 1 ? mNrof : romax + 1;
  return {&mPrimaryVertices[mROFramesPV[romin]], static_cast<gsl::span<const Vertex>::size_type>(mROFramesPV[stop_idx] - mROFramesPV[romin])};
}

template <int nLayers>
inline gsl::span<const std::array<float, 2>> TimeFrame<nLayers>::getPrimaryVerticesXAlpha(int rofId) const
{
  const int start = mROFramesPV[rofId];
  const int stop_idx = rofId >= mNrof - 1 ? mNrof : rofId + 1;
  int delta = mMultiplicityCutMask[rofId] ? mROFramesPV[stop_idx] - start : 0; // return empty span if Rof is excluded
  return {&(mPValphaX[start]), static_cast<gsl::span<const std::array<float, 2>>::size_type>(delta)};
}

template <int nLayers>
inline int TimeFrame<nLayers>::getPrimaryVerticesNum(int rofId) const
{
  return rofId < 0 ? mPrimaryVertices.size() : mROFramesPV[rofId + 1] - mROFramesPV[rofId];
}

template <int nLayers>
inline void TimeFrame<nLayers>::resetBeamXY(const float x, const float y, const float w)
{
  mBeamPos[0] = x;
  mBeamPos[1] = y;
  mBeamPosWeight = w;
}

template <int nLayers>
inline gsl::span<const int> TimeFrame<nLayers>::getROFrameClusters(int layerId) const
{
  return {&mROFramesClusters[layerId][0], static_cast<gsl::span<const int>::size_type>(mROFramesClusters[layerId].size())};
}

template <int nLayers>
inline gsl::span<Cluster> TimeFrame<nLayers>::getClustersOnLayer(int rofId, int layerId)
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mClusters[layerId][startIdx], static_cast<gsl::span<Cluster>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<const Cluster> TimeFrame<nLayers>::getClustersOnLayer(int rofId, int layerId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mClusters[layerId][startIdx], static_cast<gsl::span<const Cluster>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<uint8_t> TimeFrame<nLayers>::getUsedClustersROF(int rofId, int layerId)
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mUsedClusters[layerId][startIdx], static_cast<gsl::span<uint8_t>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<const uint8_t> TimeFrame<nLayers>::getUsedClustersROF(int rofId, int layerId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mUsedClusters[layerId][startIdx], static_cast<gsl::span<const uint8_t>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<const Cluster> TimeFrame<nLayers>::getClustersPerROFrange(int rofMin, int range, int layerId) const
{
  if (rofMin < 0 || rofMin >= mNrof) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofMin]}; // First cluster of rofMin
  int endIdx{mROFramesClusters[layerId][o2::gpu::CAMath::Min(rofMin + range, mNrof)]};
  return {&mClusters[layerId][startIdx], static_cast<gsl::span<Cluster>::size_type>(endIdx - startIdx)};
}

template <int nLayers>
inline gsl::span<const int> TimeFrame<nLayers>::getROFramesClustersPerROFrange(int rofMin, int range, int layerId) const
{
  int chkdRange{o2::gpu::CAMath::Min(range, mNrof - rofMin)};
  return {&mROFramesClusters[layerId][rofMin], static_cast<gsl::span<int>::size_type>(chkdRange)};
}

template <int nLayers>
inline gsl::span<const int> TimeFrame<nLayers>::getNClustersROFrange(int rofMin, int range, int layerId) const
{
  int chkdRange{o2::gpu::CAMath::Min(range, mNrof - rofMin)};
  return {&mNClustersPerROF[layerId][rofMin], static_cast<gsl::span<int>::size_type>(chkdRange)};
}

template <int nLayers>
inline int TimeFrame<nLayers>::getTotalClustersPerROFrange(int rofMin, int range, int layerId) const
{
  int startIdx{rofMin}; // First cluster of rofMin
  int endIdx{o2::gpu::CAMath::Min(rofMin + range, mNrof)};
  return mROFramesClusters[layerId][endIdx] - mROFramesClusters[layerId][startIdx];
}

template <int nLayers>
inline gsl::span<const int> TimeFrame<nLayers>::getIndexTablePerROFrange(int rofMin, int range, int layerId) const
{
  const int iTableSize{mIndexTableUtils.getNphiBins() * mIndexTableUtils.getNzBins() + 1};
  int chkdRange{o2::gpu::CAMath::Min(range, mNrof - rofMin)};
  return {&mIndexTables[layerId][rofMin * iTableSize], static_cast<gsl::span<int>::size_type>(chkdRange * iTableSize)};
}

template <int nLayers>
inline int TimeFrame<nLayers>::getClusterROF(int iLayer, int iCluster)
{
  return std::lower_bound(mROFramesClusters[iLayer].begin(), mROFramesClusters[iLayer].end(), iCluster + 1) - mROFramesClusters[iLayer].begin() - 1;
}

template <int nLayers>
inline gsl::span<const Cluster> TimeFrame<nLayers>::getUnsortedClustersOnLayer(int rofId, int layerId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  int startIdx{mROFramesClusters[layerId][rofId]};
  return {&mUnsortedClusters[layerId][startIdx], static_cast<gsl::span<Cluster>::size_type>(mROFramesClusters[layerId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<int> TimeFrame<nLayers>::getIndexTable(int rofId, int layer)
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  const int tableSize = mIndexTableUtils.getNphiBins() * mIndexTableUtils.getNzBins() + 1;
  return {&mIndexTables[layer][rofId * tableSize], static_cast<gsl::span<int>::size_type>(tableSize)};
}

template <int nLayers>
template <typename... T>
void TimeFrame<nLayers>::addClusterToLayer(int layer, T&&... values)
{
  mUnsortedClusters[layer].emplace_back(std::forward<T>(values)...);
}

template <int nLayers>
template <typename... T>
void TimeFrame<nLayers>::addTrackingFrameInfoToLayer(int layer, T&&... values)
{
  mTrackingFrameInfo[layer].emplace_back(std::forward<T>(values)...);
}

template <int nLayers>
inline gsl::span<uint8_t> TimeFrame<nLayers>::getUsedClusters(const int layer)
{
  return {&mUsedClusters[layer][0], static_cast<gsl::span<uint8_t>::size_type>(mUsedClusters[layer].size())};
}

template <int nLayers>
inline void TimeFrame<nLayers>::initialiseRoadLabels()
{
  mRoadLabels.clear();
  mRoadLabels.resize(mRoads.size());
}

template <int nLayers>
inline void TimeFrame<nLayers>::setRoadLabel(int i, const unsigned long long& lab, bool fake)
{
  mRoadLabels[i].first = lab;
  mRoadLabels[i].second = fake;
}

template <int nLayers>
inline gsl::span<int> TimeFrame<nLayers>::getNTrackletsCluster(int rofId, int combId)
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  auto startIdx{mROFramesClusters[1][rofId]};
  return {&mNTrackletsPerCluster[combId][startIdx], static_cast<gsl::span<int>::size_type>(mROFramesClusters[1][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<int> TimeFrame<nLayers>::getExclusiveNTrackletsCluster(int rofId, int combId)
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  auto clusStartIdx{mROFramesClusters[1][rofId]};

  return {&mNTrackletsPerClusterSum[combId][clusStartIdx], static_cast<gsl::span<int>::size_type>(mROFramesClusters[1][rofId + 1] - clusStartIdx)};
}

template <int nLayers>
inline gsl::span<Tracklet> TimeFrame<nLayers>::getFoundTracklets(int rofId, int combId)
{
  if (rofId < 0 || rofId >= mNrof || mTracklets[combId].empty()) {
    return {};
  }
  auto startIdx{mNTrackletsPerROF[combId][rofId]};
  return {&mTracklets[combId][startIdx], static_cast<gsl::span<Tracklet>::size_type>(mNTrackletsPerROF[combId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<const Tracklet> TimeFrame<nLayers>::getFoundTracklets(int rofId, int combId) const
{
  if (rofId < 0 || rofId >= mNrof) {
    return {};
  }
  auto startIdx{mNTrackletsPerROF[combId][rofId]};
  return {&mTracklets[combId][startIdx], static_cast<gsl::span<Tracklet>::size_type>(mNTrackletsPerROF[combId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline gsl::span<const MCCompLabel> TimeFrame<nLayers>::getLabelsFoundTracklets(int rofId, int combId) const
{
  if (rofId < 0 || rofId >= mNrof || !hasMCinformation()) {
    return {};
  }
  auto startIdx{mNTrackletsPerROF[combId][rofId]};
  return {&mTrackletLabels[combId][startIdx], static_cast<gsl::span<Tracklet>::size_type>(mNTrackletsPerROF[combId][rofId + 1] - startIdx)};
}

template <int nLayers>
inline int TimeFrame<nLayers>::getTotalClusters() const
{
  size_t totalClusters{0};
  for (const auto& clusters : mUnsortedClusters) {
    totalClusters += clusters.size();
  }
  return int(totalClusters);
}

template <int nLayers>
inline int TimeFrame<nLayers>::getNumberOfClusters() const
{
  int nClusters = 0;
  for (const auto& layer : mClusters) {
    nClusters += layer.size();
  }
  return nClusters;
}

template <int nLayers>
inline int TimeFrame<nLayers>::getNumberOfCells() const
{
  int nCells = 0;
  for (const auto& layer : mCells) {
    nCells += layer.size();
  }
  return nCells;
}

template <int nLayers>
inline int TimeFrame<nLayers>::getNumberOfTracklets() const
{
  int nTracklets = 0;
  for (const auto& layer : mTracklets) {
    nTracklets += layer.size();
  }
  return nTracklets;
}

template <int nLayers>
inline int TimeFrame<nLayers>::getNumberOfNeighbours() const
{
  int n{0};
  for (const auto& l : mCellsNeighbours) {
    n += l.size();
  }
  return n;
}

template <int nLayers>
inline size_t TimeFrame<nLayers>::getNumberOfTracks() const
{
  int nTracks = 0;
  for (const auto& t : mTracks) {
    nTracks += t.size();
  }
  return nTracks;
}

template <int nLayers>
inline size_t TimeFrame<nLayers>::getNumberOfUsedClusters() const
{
  size_t nClusters = 0;
  for (const auto& layer : mUsedClusters) {
    nClusters += std::count(layer.begin(), layer.end(), true);
  }
  return nClusters;
}

template <int nLayers>
inline void TimeFrame<nLayers>::insertPastVertex(const Vertex& vertex, const int iteration)
{
  int rofId = vertex.getTimeStamp().getTimeStamp();
  mPrimaryVertices.insert(mPrimaryVertices.begin() + mROFramesPV[rofId], vertex);
  for (int i = rofId + 1; i < mROFramesPV.size(); ++i) {
    mROFramesPV[i]++;
  }
  mTotVertPerIteration[iteration]++;
}

} // namespace its
} // namespace o2

#endif

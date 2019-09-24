// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file DeviceStoreVertexer.h
/// \brief This class serves as memory interface for GPU vertexer. It will access needed data structures from devicestore apis.
///        routines as static as possible.
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_
#define O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/ClusterLines.h"
#include "ITStrackingCUDA/Array.h"
#include "ITStrackingCUDA/UniquePointer.h"
#include "ITStrackingCUDA/Vector.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{
namespace GPU
{
enum class TrackletingLayerOrder {
  fromInnermostToMiddleLayer,
  fromMiddleToOuterLayer
};

enum class VertexerLayerName {
  innermostLayer,
  middleLayer,
  outerLayer
};

typedef TrackletingLayerOrder Order;
class DeviceStoreVertexerGPU final
{
 public:
  DeviceStoreVertexerGPU();
  ~DeviceStoreVertexerGPU() = default;

  UniquePointer<DeviceStoreVertexerGPU> initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>&,
                                                   const std::array<std::array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                                                                    constants::its::LayersNumberVertexer>&);

  // RO APIs
  GPUd() const Array<Vector<Cluster>, constants::its::LayersNumberVertexer>& getClusters()
  {
    return mClusters;
  }
  GPUd() const Vector<int>& getIndexTable(const VertexerLayerName);
  GPUhd() const VertexerStoreConfigurationGPU& getConfig() { return mGPUConf; }

  // Writable APIs
  GPUd() Vector<Tracklet>& getDuplets01() { return mDuplets01; }
  GPUd() Vector<Tracklet>& getDuplets12() { return mDuplets12; }
  GPUd() Vector<Line>& getLines() { return mTracklets; }
  GPUd() Vector<int>& getNFoundLines() { return mNFoundLines; }
#ifdef _ALLOW_DEBUG_TREES_ITS_
  GPUd() Array<Vector<int>, 2>& getDupletIndices()
  {
    return mDupletIndices;
  }
#endif
  GPUhd() Vector<int>& getNFoundTracklets(Order order)
  {
    return mNFoundDuplets[static_cast<int>(order)];
  }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  GPUh() void updateDuplets(const Order, std::vector<Tracklet>&);
  GPUh() void updateFoundDuplets(const Order, std::vector<int>&);
  GPUh() std::vector<int> getNFoundTrackletsFromGPU(const Order);
  GPUh() std::vector<Tracklet> getRawDupletsFromGPU(const Order);
  GPUh() std::vector<Tracklet> getDupletsFromGPU(const Order);
  GPUh() std::vector<Line> getRawLinesFromGPU();
  GPUh() std::vector<std::array<int, 2>> getDupletIndicesFromGPU();
  GPUh() std::vector<int> getNFoundLinesFromGPU();
#endif
  // This is temporary kept outside the debug region, since it is used to bridge data skimmed on GPU to final vertex calculation on CPU.
  // Eventually, all the vertexing will be done on GPU, so this will become a debug API.
  GPUh() std::vector<Line> getLinesFromGPU();

 private:
  Vector<int> mSizes;
  VertexerStoreConfigurationGPU mGPUConf;
  Array<Vector<Cluster>, constants::its::LayersNumberVertexer> mClusters;
  Array<Vector<int>, constants::its::LayersNumberVertexer - 1> mNFoundDuplets;
  Vector<int> mNFoundLines;
  Vector<Tracklet> mDuplets01;
  Vector<Tracklet> mDuplets12;
  Vector<Line> mTracklets;
  Array<Vector<int>, 2> mIndexTables;

#ifdef _ALLOW_DEBUG_TREES_ITS_
  Array<Vector<int>, 2> mDupletIndices;
#endif
};

#ifdef _ALLOW_DEBUG_TREES_ITS_
inline std::vector<int> DeviceStoreVertexerGPU::getNFoundTrackletsFromGPU(const Order order)
{
  // Careful: this might lead to large allocations, use debug-purpose only
  std::vector<int> sizes;
  sizes.resize(constants::its::LayersNumberVertexer);
  mSizes.copyIntoSizedVector(sizes);
  std::vector<int> nFoundDuplets;
  nFoundDuplets.resize(sizes[1]);

  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].copyIntoSizedVector(nFoundDuplets);
  } else {
    mNFoundDuplets[1].copyIntoSizedVector(nFoundDuplets);
  }

  return nFoundDuplets;
}

inline std::vector<Tracklet> DeviceStoreVertexerGPU::getRawDupletsFromGPU(const Order order)
{
  // Careful: this might lead to large allocations, use debug-purpose only
  std::vector<int> sizes;
  sizes.resize(constants::its::LayersNumberVertexer);
  mSizes.copyIntoSizedVector(sizes);
  std::vector<Tracklet> tmpDuplets;
  tmpDuplets.resize(static_cast<size_t>(mGPUConf.dupletsCapacity));
  std::vector<int> nFoundDuplets;
  nFoundDuplets.resize(sizes[1]);

  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].copyIntoSizedVector(nFoundDuplets);
    mDuplets01.copyIntoSizedVector(tmpDuplets);
  } else {
    mDuplets12.copyIntoSizedVector(tmpDuplets);
    mNFoundDuplets[1].copyIntoSizedVector(nFoundDuplets);
  }

  return tmpDuplets;
}

inline std::vector<Tracklet> DeviceStoreVertexerGPU::getDupletsFromGPU(const Order order)
{
  // Careful: this might lead to large allocations, use debug-purpose only
  std::vector<int> sizes;
  sizes.resize(constants::its::LayersNumberVertexer);
  mSizes.copyIntoSizedVector(sizes);
  std::vector<Tracklet> tmpDuplets;
  tmpDuplets.resize(static_cast<size_t>(mGPUConf.dupletsCapacity));
  std::vector<int> nFoundDuplets;
  nFoundDuplets.resize(sizes[1]);
  std::vector<Tracklet> shrinkedDuplets;

  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].copyIntoSizedVector(nFoundDuplets);
    mDuplets01.copyIntoSizedVector(tmpDuplets);
  } else {
    mDuplets12.copyIntoSizedVector(tmpDuplets);
    mNFoundDuplets[1].copyIntoSizedVector(nFoundDuplets);
  }

  for (int iCluster{0}; iCluster < sizes[1]; ++iCluster) {
    const int stride{iCluster * mGPUConf.maxTrackletsPerCluster};
    for (int iDuplet{0}; iDuplet < nFoundDuplets[iCluster]; ++iDuplet) {
      shrinkedDuplets.push_back(tmpDuplets[stride + iDuplet]);
    }
  }
  return shrinkedDuplets;
}

inline std::vector<std::array<int, 2>> DeviceStoreVertexerGPU::getDupletIndicesFromGPU()
{
  // Careful: this might lead to large allocations, use debug-purpose only.
  std::array<std::vector<int>, 2> allowedLines;
  std::vector<std::array<int, 2>> allowedPairIndices;
  allowedPairIndices.reserve(mGPUConf.processedTrackletsCapacity);
  for (int iAllowed{0}; iAllowed < 2; ++iAllowed) {
    allowedLines[iAllowed].resize(mGPUConf.processedTrackletsCapacity);
    mDupletIndices[iAllowed].resize(mGPUConf.processedTrackletsCapacity);
    mDupletIndices[iAllowed].copyIntoSizedVector(allowedLines[iAllowed]);
  }
  for (size_t iPair{0}; iPair < allowedLines[0].size(); ++iPair) {
    allowedPairIndices.emplace_back(std::array<int, 2>{allowedLines[0][iPair], allowedLines[1][iPair]});
  }
  return allowedPairIndices;
}

inline void DeviceStoreVertexerGPU::updateDuplets(const Order order, std::vector<Tracklet>& duplets)
{
  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mDuplets01.reset(duplets.data(), static_cast<int>(duplets.size()));
  } else {
    mDuplets12.reset(duplets.data(), static_cast<int>(duplets.size()));
  }
}

inline void DeviceStoreVertexerGPU::updateFoundDuplets(const Order order, std::vector<int>& nDuplets)
{
  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].reset(nDuplets.data(), static_cast<int>(nDuplets.size()));
  } else {
    mNFoundDuplets[1].reset(nDuplets.data(), static_cast<int>(nDuplets.size()));
  }
}

inline std::vector<Line> DeviceStoreVertexerGPU::getRawLinesFromGPU()
{
  std::vector<Line> lines;
  lines.resize(mGPUConf.processedTrackletsCapacity);
  mTracklets.copyIntoSizedVector(lines);

  return lines;
}

std::vector<int> DeviceStoreVertexerGPU::getNFoundLinesFromGPU()
{
  std::vector<int> nFoundLines;
  nFoundLines.resize(mGPUConf.clustersPerLayerCapacity);
  mNFoundLines.copyIntoSizedVector(nFoundLines);

  return nFoundLines;
}
#endif
// This is temporary kept outside the debug region, since it is used to bridge data skimmed on GPU to final vertex calculation on CPU.
// Eventually, all the vertexing will be done on GPU, so this will become a debug API.

inline std::vector<Line> DeviceStoreVertexerGPU::getLinesFromGPU()
{
  std::vector<Line> lines;
  std::vector<Line> tmpLines;
  std::vector<int> nFoundLines;
  std::vector<int> sizes;
  sizes.resize(constants::its::LayersNumberVertexer);
  tmpLines.resize(mGPUConf.processedTrackletsCapacity);
  nFoundLines.resize(mGPUConf.clustersPerLayerCapacity);
  mSizes.copyIntoSizedVector(sizes);
  mTracklets.copyIntoSizedVector(tmpLines);
  mNFoundLines.copyIntoSizedVector(nFoundLines);
  for (int iCluster{0}; iCluster < sizes[1]; ++iCluster) {
    const int stride{iCluster * mGPUConf.maxTrackletsPerCluster};
    for (int iLine{0}; iLine < nFoundLines[iCluster]; ++iLine) {
      lines.push_back(tmpLines[stride + iLine]);
    }
  }

  return lines;
}

} // namespace GPU
} // namespace its
} // namespace o2
#endif //O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_

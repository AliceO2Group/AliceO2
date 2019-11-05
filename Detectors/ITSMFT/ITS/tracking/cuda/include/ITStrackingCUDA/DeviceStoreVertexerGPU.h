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

#include <cub/cub.cuh>

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/ClusterLines.h"
#include "ITStrackingCUDA/Array.h"
#include "ITStrackingCUDA/ClusterLinesGPU.h"
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
  GPUhd() VertexerStoreConfigurationGPU& getConfig() { return mGPUConf; }

  // Writable APIs
  GPUd() Vector<Tracklet>& getDuplets01() { return mDuplets01; }
  GPUd() Vector<Tracklet>& getDuplets12() { return mDuplets12; }
  GPUd() Vector<Line>& getLines() { return mTracklets; }
  GPUd() Vector<float>& getBeamPosition() { return mBeamPosition; }
  GPUhd() Vector<GPUVertex>& getVertices() { return mGPUVertices; }
  GPUhd() Vector<int>& getNFoundLines() { return mNFoundLines; }
  GPUhd() Vector<int>& getNExclusiveFoundLines() { return mNExclusiveFoundLines; }
  GPUhd() Vector<int>& getCUBTmpBuffer() { return mCUBTmpBuffer; }
  GPUhd() Vector<float>& getXYCentroids() { return mXYCentroids; }
  GPUhd() Vector<float>& getZCentroids() { return mZCentroids; }
  GPUhd() Array<Vector<int>, 3>& getHistogramXYZ() { return mHistogramXYZ; }
  GPUhd() Vector<cub::KeyValuePair<int, int>>& getTmpVertexPositionBins() { return mTmpVertexPositionBins; }

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
  GPUh() std::array<std::vector<int>, 2> getHistogramXYFromGPU();
  GPUh() std::vector<int> getHistogramZFromGPU();
#endif
  // This is temporary kept outside the debug region, since it is used to bridge data skimmed on GPU to final vertex calculation on CPU.
  // Eventually, all the vertexing will be done on GPU, so this will become a debug API.
  GPUh() std::vector<Line> getLinesFromGPU();

 private:
  VertexerStoreConfigurationGPU mGPUConf;
  Array<Vector<Cluster>, constants::its::LayersNumberVertexer> mClusters;
  Vector<Line> mTracklets;
  Array<Vector<int>, 2> mIndexTables;
  Vector<GPUVertex> mGPUVertices;

  // service buffers
  Vector<int> mNFoundLines;
  Vector<int> mNExclusiveFoundLines;
  Vector<Tracklet> mDuplets01;
  Vector<Tracklet> mDuplets12;
  Array<Vector<int>, constants::its::LayersNumberVertexer - 1> mNFoundDuplets;
  Vector<int> mCUBTmpBuffer;
  Vector<float> mXYCentroids;
  Vector<float> mZCentroids;
  Vector<float> mBeamPosition;
  Array<Vector<int>, 3> mHistogramXYZ;
  Vector<cub::KeyValuePair<int, int>> mTmpVertexPositionBins;

#ifdef _ALLOW_DEBUG_TREES_ITS_
  Array<Vector<int>, 2> mDupletIndices;
  Vector<int> mSizes;
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
  int nLines = getNExclusiveFoundLines().getElementFromDevice(mClusters[1].getSizeFromDevice() - 1) + getNFoundLines().getElementFromDevice(mClusters[1].getSizeFromDevice() - 1);
  allowedPairIndices.reserve(nLines);
  for (int iAllowed{0}; iAllowed < 2; ++iAllowed) {
    allowedLines[iAllowed].resize(nLines);
    mDupletIndices[iAllowed].resize(nLines);
    mDupletIndices[iAllowed].copyIntoSizedVector(allowedLines[iAllowed]);
  }
  for (size_t iPair{0}; iPair < allowedLines[0].size(); ++iPair) {
    allowedPairIndices.emplace_back(std::array<int, 2>{allowedLines[0][iPair], allowedLines[1][iPair]});
  }
  return allowedPairIndices;
}

inline std::array<std::vector<int>, 2> DeviceStoreVertexerGPU::getHistogramXYFromGPU()
{
  std::array<std::vector<int>, 2> histoXY;
  for (int iHisto{0}; iHisto < 2; ++iHisto) {
    histoXY[iHisto].resize(mGPUConf.nBinsXYZ[iHisto] - 1);
    mHistogramXYZ[iHisto].copyIntoSizedVector(histoXY[iHisto]);
  }

  return histoXY;
}

inline std::vector<int> DeviceStoreVertexerGPU::getHistogramZFromGPU()
{
  std::vector<int> histoZ;
  histoZ.resize(mGPUConf.nBinsXYZ[2] - 1);
  std::cout << "Size of dest vector to be refined" << std::endl;
  mHistogramXYZ[2].copyIntoSizedVector(histoZ);

  return histoZ;
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
  tmpLines.resize(mGPUConf.processedTrackletsCapacity);
  mTracklets.copyIntoSizedVector(tmpLines);
  for (auto& line : tmpLines) {
    if (line.isEmpty) {
      break;
    }
    lines.push_back(line);
  }
  return lines;
}

} // namespace GPU
} // namespace its
} // namespace o2
#endif //O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_

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
/// \file DeviceStoreVertexerHIP.h
/// \brief This class serves as memory interface for GPU vertexer. It will access needed data structures from devicestore apis.
///        routines as static as possible.
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_INCLUDE_DEVICESTOREVERTEXER_HIP_H_
#define O2_ITS_TRACKING_INCLUDE_DEVICESTOREVERTEXER_HIP_H_

#include <hipcub/hipcub.hpp>

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/ClusterLines.h"
#include "ITStrackingHIP/ArrayHIP.h"
#include "ITStrackingHIP/ClusterLinesHIP.h"
#include "ITStrackingHIP/UniquePointerHIP.h"
#include "ITStrackingHIP/VectorHIP.h"
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
class DeviceStoreVertexerHIP final
{
 public:
  DeviceStoreVertexerHIP();
  ~DeviceStoreVertexerHIP() = default;

  UniquePointer<DeviceStoreVertexerHIP> initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>&,
                                                   const std::array<std::array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                                                                    constants::its::LayersNumberVertexer>&);

  // RO APIs
  GPUd() const ArrayHIP<VectorHIP<Cluster>, constants::its::LayersNumberVertexer>& getClusters()
  {
    return mClusters;
  }
  GPUd() const VectorHIP<int>& getIndexTable(const VertexerLayerName);
  GPUhd() VertexerStoreConfigurationGPU& getConfig() { return mGPUConf; }

  // Writable APIs
  GPUd() VectorHIP<Tracklet>& getDuplets01() { return mDuplets01; }
  GPUd() VectorHIP<Tracklet>& getDuplets12() { return mDuplets12; }
  GPUd() VectorHIP<Line>& getLines() { return mTracklets; }
  GPUd() VectorHIP<float>& getBeamPosition() { return mBeamPosition; }
  GPUhd() VectorHIP<GPUVertex>& getVertices() { return mGPUVertices; }
  GPUhd() VectorHIP<int>& getNFoundLines() { return mNFoundLines; }
  GPUhd() VectorHIP<int>& getNExclusiveFoundLines() { return mNExclusiveFoundLines; }
  GPUhd() VectorHIP<int>& getCUBTmpBuffer() { return mCUBTmpBuffer; }
  GPUhd() VectorHIP<float>& getXYCentroids() { return mXYCentroids; }
  GPUhd() VectorHIP<float>& getZCentroids() { return mZCentroids; }
  GPUhd() ArrayHIP<VectorHIP<int>, 3>& getHistogramXYZ() { return mHistogramXYZ; }
  GPUhd() VectorHIP<hipcub::KeyValuePair<int, int>>& getTmpVertexPositionBins() { return mTmpVertexPositionBins; }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  GPUd() ArrayHIP<VectorHIP<int>, 2>& getDupletIndices()
  {
    return mDupletIndices;
  }
#endif
  GPUhd() VectorHIP<int>& getNFoundTracklets(Order order)
  {
    return mNFoundDuplets[static_cast<int>(order)];
  }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  /*GPUh()*/ void updateDuplets(const Order, std::vector<Tracklet>&);
  /*GPUh()*/ void updateFoundDuplets(const Order, std::vector<int>&);
  /*GPUh()*/ std::vector<int> getNFoundTrackletsFromGPU(const Order);
  /*GPUh()*/ std::vector<Tracklet> getRawDupletsFromGPU(const Order);
  /*GPUh()*/ std::vector<Tracklet> getDupletsFromGPU(const Order);
  /*GPUh()*/ std::vector<Line> getRawLinesFromGPU();
  /*GPUh()*/ std::vector<std::array<int, 2>> getDupletIndicesFromGPU();
  /*GPUh()*/ std::vector<int> getNFoundLinesFromGPU();
  /*GPUh()*/ std::array<std::vector<int>, 2> getHistogramXYFromGPU();
  /*GPUh()*/ std::vector<int> getHistogramZFromGPU();
  /*GPUh()*/ std::vector<Line> getLinesFromGPU();
#endif

 private:
  VertexerStoreConfigurationGPU mGPUConf;
  ArrayHIP<VectorHIP<Cluster>, constants::its::LayersNumberVertexer> mClusters;
  VectorHIP<Line> mTracklets;
  ArrayHIP<VectorHIP<int>, 2> mIndexTables;
  VectorHIP<GPUVertex> mGPUVertices;

  // service buffers
  VectorHIP<int> mNFoundLines;
  VectorHIP<int> mNExclusiveFoundLines;
  VectorHIP<Tracklet> mDuplets01;
  VectorHIP<Tracklet> mDuplets12;
  ArrayHIP<VectorHIP<int>, constants::its::LayersNumberVertexer - 1> mNFoundDuplets;
  VectorHIP<int> mCUBTmpBuffer;
  VectorHIP<float> mXYCentroids;
  VectorHIP<float> mZCentroids;
  VectorHIP<float> mBeamPosition;
  ArrayHIP<VectorHIP<int>, 3> mHistogramXYZ;
  VectorHIP<hipcub::KeyValuePair<int, int>> mTmpVertexPositionBins;

#ifdef _ALLOW_DEBUG_TREES_ITS_
  ArrayHIP<VectorHIP<int>, 2> mDupletIndices;
  VectorHIP<int> mSizes;
#endif
};

#ifdef _ALLOW_DEBUG_TREES_ITS_
inline std::vector<int> DeviceStoreVertexerHIP::getNFoundTrackletsFromGPU(const Order order)
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

inline std::vector<Tracklet> DeviceStoreVertexerHIP::getRawDupletsFromGPU(const Order order)
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

inline std::vector<Tracklet> DeviceStoreVertexerHIP::getDupletsFromGPU(const Order order)
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

inline std::vector<std::array<int, 2>> DeviceStoreVertexerHIP::getDupletIndicesFromGPU()
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

inline std::array<std::vector<int>, 2> DeviceStoreVertexerHIP::getHistogramXYFromGPU()
{
  std::array<std::vector<int>, 2> histoXY;
  for (int iHisto{0}; iHisto < 2; ++iHisto) {
    histoXY[iHisto].resize(mGPUConf.nBinsXYZ[iHisto] - 1);
    mHistogramXYZ[iHisto].copyIntoSizedVector(histoXY[iHisto]);
  }

  return histoXY;
}

inline std::vector<int> DeviceStoreVertexerHIP::getHistogramZFromGPU()
{
  std::vector<int> histoZ;
  histoZ.resize(mGPUConf.nBinsXYZ[2] - 1);
  std::cout << "Size of dest vector to be refined" << std::endl;
  mHistogramXYZ[2].copyIntoSizedVector(histoZ);

  return histoZ;
}

inline void DeviceStoreVertexerHIP::updateDuplets(const Order order, std::vector<Tracklet>& duplets)
{
  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mDuplets01.reset(duplets.data(), static_cast<int>(duplets.size()));
  } else {
    mDuplets12.reset(duplets.data(), static_cast<int>(duplets.size()));
  }
}

inline void DeviceStoreVertexerHIP::updateFoundDuplets(const Order order, std::vector<int>& nDuplets)
{
  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].reset(nDuplets.data(), static_cast<int>(nDuplets.size()));
  } else {
    mNFoundDuplets[1].reset(nDuplets.data(), static_cast<int>(nDuplets.size()));
  }
}

inline std::vector<Line> DeviceStoreVertexerHIP::getRawLinesFromGPU()
{
  std::vector<Line> lines;
  lines.resize(mGPUConf.processedTrackletsCapacity);
  mTracklets.copyIntoSizedVector(lines);

  return lines;
}

inline std::vector<int> DeviceStoreVertexerHIP::getNFoundLinesFromGPU()
{
  std::vector<int> nFoundLines;
  nFoundLines.resize(mGPUConf.clustersPerLayerCapacity);
  mNFoundLines.copyIntoSizedVector(nFoundLines);

  return nFoundLines;
}

inline std::vector<Line> DeviceStoreVertexerHIP::getLinesFromGPU()
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
#endif
} // namespace GPU
} // namespace its
} // namespace o2
#endif //O2_ITS_TRACKING_INCLUDE_DEVICESTOREVERTEXER_HIP_H_

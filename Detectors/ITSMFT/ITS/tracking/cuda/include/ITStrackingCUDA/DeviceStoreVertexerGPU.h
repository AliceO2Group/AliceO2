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
  GPUd() const Array<Vector<Cluster>, constants::its::LayersNumberVertexer>& getClusters() { return mClusters; }
  GPUd() const Vector<int>& getIndexTable(const VertexerLayerName);
  GPUd() const VertexerStoreConfigurationGPU& getConfig() { return mGPUConf; }

  // Writable APIs
  GPUd() Vector<Tracklet>& getDuplets01() { return mDuplets01; }
  GPUd() Vector<Tracklet>& getDuplets12() { return mDuplets12; }
  GPUd() Vector<Line>& getLines() { return mTracklets; }
  GPUd() Vector<int>& getNFoundLines() { return mNFoundLines; }
  GPUd() Array<Vector<int>, 2>& getDupletIndices()
  {
    return mDupletIndices;
  }
  GPUhd() Vector<int>& getNFoundTracklets(Order order)
  {
    return mNFoundDuplets[static_cast<int>(order)];
  }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  GPUh() std::vector<int> getNFoundTrackletsFromGPU(const Order);
  GPUh() std::vector<Tracklet> getRawDupletsFromGPU(const Order);
  GPUh() std::vector<Tracklet> getDupletsFromGPU(const Order);
  GPUh() std::vector<Line> getLinesFromGPU();
  GPUh() std::array<std::vector<int>, 2> getDupletIndicesFromGPU();
#endif

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
  std::vector<int> sizes{constants::its::LayersNumberVertexer};
  mSizes.copyIntoVector(sizes, constants::its::LayersNumberVertexer);
  std::vector<int> nFoundDuplets{sizes[1]};

  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].copyIntoVector(nFoundDuplets, sizes[1]);
  } else {
    mNFoundDuplets[1].copyIntoVector(nFoundDuplets, sizes[1]);
  }

  return nFoundDuplets;
}

inline std::vector<Tracklet> DeviceStoreVertexerGPU::getRawDupletsFromGPU(const Order order)
{
  // Careful: this might lead to large allocations, use debug-purpose only
  std::vector<int> sizes{constants::its::LayersNumberVertexer};
  mSizes.copyIntoVector(sizes, constants::its::LayersNumberVertexer);
  std::vector<Tracklet> tmpDuplets{static_cast<size_t>(mGPUConf.dupletsCapacity)};
  std::vector<int> nFoundDuplets{sizes[1]};

  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].copyIntoVector(nFoundDuplets, sizes[1]);
    mDuplets01.copyIntoVector(tmpDuplets, tmpDuplets.size());
  } else {
    mDuplets12.copyIntoVector(tmpDuplets, tmpDuplets.size());
    mNFoundDuplets[1].copyIntoVector(nFoundDuplets, sizes[1]);
  }

  return tmpDuplets;
}

inline std::vector<Tracklet> DeviceStoreVertexerGPU::getDupletsFromGPU(const Order order)
{
  // Careful: this might lead to large allocations, use debug-purpose only
  std::vector<int> sizes{constants::its::LayersNumberVertexer};
  mSizes.copyIntoVector(sizes, constants::its::LayersNumberVertexer);
  std::vector<Tracklet> tmpDuplets{static_cast<size_t>(mGPUConf.dupletsCapacity)};
  std::vector<int> nFoundDuplets{sizes[1]};
  std::vector<Tracklet> shrinkedDuplets;

  if (order == GPU::Order::fromInnermostToMiddleLayer) {
    mNFoundDuplets[0].copyIntoVector(nFoundDuplets, sizes[1]);
    mDuplets01.copyIntoVector(tmpDuplets, tmpDuplets.size());
  } else {
    mDuplets12.copyIntoVector(tmpDuplets, tmpDuplets.size());
    mNFoundDuplets[1].copyIntoVector(nFoundDuplets, sizes[1]);
  }

  for (int iCluster{0}; iCluster < sizes[1]; ++iCluster) {
    const int stride{iCluster * mGPUConf.maxTrackletsPerCluster};
    for (int iDuplet{0}; iDuplet < nFoundDuplets[iCluster]; ++iDuplet) {
      shrinkedDuplets.push_back(tmpDuplets[stride + iDuplet]);
    }
  }
  return shrinkedDuplets;
}

inline std::array<std::vector<int>, 2> DeviceStoreVertexerGPU::getDupletIndicesFromGPU()
{
  // Careful: this might lead to large allocations, use debug-purpose only.
  std::array<std::vector<int>, 2> allowedLines;
  for (int iAllowed{0}; iAllowed < 2; ++iAllowed) {
    allowedLines[iAllowed].resize(mGPUConf.processedTrackletsCapacity);
    mDupletIndices[iAllowed].copyIntoVector(allowedLines[iAllowed], allowedLines[iAllowed].size());
  }
  return allowedLines;
}

inline std::vector<Line> DeviceStoreVertexerGPU::getLinesFromGPU()
{
  std::vector<Line> lines;
  lines.resize(mGPUConf.processedTrackletsCapacity);
  mTracklets.copyIntoVector(lines, lines.size());

  return lines;
}
#endif
} // namespace GPU
} // namespace its
} // namespace o2
#endif //O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_

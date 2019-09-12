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

struct VertexerStoreConfigurationGPU {
  // o2::its::GPU::Vector constructor requires signed size for initialisation
  int dupletsCapacity = 40e6;
  int processedTrackletsCapacity = 40e6;
  int clustersPerLayerCapacity = 50e3;
  int maxTrackletsPerCluster = 1e3;
};

class DeviceStoreVertexerGPU final
{
 public:
  DeviceStoreVertexerGPU();
  ~DeviceStoreVertexerGPU() = default;

  UniquePointer<DeviceStoreVertexerGPU> initialise(const std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>&,
                                                   const std::array<std::array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                                                                    constants::its::LayersNumberVertexer>&);

  // RO APIs
  GPUhd() const Array<Vector<Cluster>, constants::its::LayersNumberVertexer>& getClusters() { return mClusters; }
  GPUhd() const Vector<int>& getIndexTable(const VertexerLayerName);
  GPUhd() const VertexerStoreConfigurationGPU& getConfig() { return mGPUConf; }

  // Writable APIs
  GPUhd() Vector<Tracklet>& getDuplets01() { return mDuplets01; }
  GPUhd() Vector<Tracklet>& getDuplets12() { return mDuplets12; }
  GPUhd() Vector<Line>& getLines() { return mTracklets; }
  GPUhd() Vector<int>& getNFoundLines() { return mNFoundLines; }
  GPUhd() Vector<int>& getNFoundTracklets(GPU::TrackletingLayerOrder order) { return mNFoundTracklets[static_cast<int>(order)]; }
  GPUd() int getFlag() { return mFlag; }
  GPUh() std::vector<Tracklet> getDupletsfromGPU(const TrackletingLayerOrder);
  GPUh() std::vector<Line> getLinesfromGPU();

 private:
  int mFlag;
  int mSizes[3];
  VertexerStoreConfigurationGPU mGPUConf;
  Array<Vector<Cluster>, constants::its::LayersNumberVertexer> mClusters;
  Array<Vector<int>, constants::its::LayersNumberVertexer - 1> mNFoundTracklets;
  Vector<int> mNFoundLines;
  Vector<Tracklet> mDuplets01;
  Vector<Tracklet> mDuplets12;
  Vector<Line> mTracklets;
  Array<Vector<int>, 2> mIndexTables;
};

inline const Vector<int>& DeviceStoreVertexerGPU::getIndexTable(const VertexerLayerName layer)
{
  if (layer == VertexerLayerName::innermostLayer) {
    return mIndexTables[0];
  }
  return mIndexTables[1];
}

} // namespace GPU
} // namespace its
} // namespace o2
#endif //O2_ITS_TRACKING_DEVICE_STORE_VERTEXER_GPU_H_

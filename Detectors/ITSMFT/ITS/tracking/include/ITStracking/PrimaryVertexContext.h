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
/// \file PrimaryVertexContext.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXT_H_
#define TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXT_H_

#include <algorithm>
#include <array>
#include <iosfwd>
#include <vector>

#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include "ITStracking/Cell.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/Road.h"
#include "ITStracking/Tracklet.h"

#if TRACKINGITSU_GPU_MODE
#include "ITStracking/gpu/PrimaryVertexContext.h"
#include "ITStracking/gpu/UniquePointer.h"
#endif

namespace o2
{
namespace ITS
{

class PrimaryVertexContext final
{
 public:
  PrimaryVertexContext();

  PrimaryVertexContext(const PrimaryVertexContext&) = delete;
  PrimaryVertexContext& operator=(const PrimaryVertexContext&) = delete;

  void initialise(const MemoryParameters& memParam, const ROframe& event, const int pvIndex, const int iteration);
  const float3& getPrimaryVertex() const;
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& getClusters();
  std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad>& getCells();
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1>& getCellsLookupTable();
  std::array<std::vector<std::vector<int>>, Constants::ITS::CellsPerRoad - 1>& getCellsNeighbours();
  std::vector<Road>& getRoads();
  std::vector<TrackITS>& getTracks();
  std::vector<MCCompLabel>& getTrackLabels();

  bool isClusterUsed(int layer, int clusterId) const;
  void markUsedCluster(int layer, int clusterId);

#if TRACKINGITSU_GPU_MODE
  GPU::PrimaryVertexContext& getDeviceContext();
  GPU::Array<GPU::Vector<Cluster>, Constants::ITS::LayersNumber>& getDeviceClusters();
  GPU::Array<GPU::Vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& getDeviceTracklets();
  GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>& getDeviceTrackletsLookupTable();
  GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>& getDeviceTrackletsPerClustersTable();
  GPU::Array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad>& getDeviceCells();
  GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad - 1>& getDeviceCellsLookupTable();
  GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad - 1>& getDeviceCellsPerTrackletTable();
  std::array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>& getTempTableArray();
  std::array<GPU::Vector<Tracklet>, Constants::ITS::CellsPerRoad>& getTempTrackletArray();
  std::array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad - 1>& getTempCellArray();
  void updateDeviceContext();
#else
  std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
             Constants::ITS::TrackletsPerRoad>&
    getIndexTables();
  std::array<std::vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& getTracklets();
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad>& getTrackletsLookupTable();
#endif

 private:
  float3 mPrimaryVertex;
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumber> mClusters;
  std::array<std::vector<bool>, Constants::ITS::LayersNumber> mUsedClusters;
  std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad> mCells;
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1> mCellsLookupTable;
  std::array<std::vector<std::vector<int>>, Constants::ITS::CellsPerRoad - 1> mCellsNeighbours;
  std::vector<Road> mRoads;
  std::vector<TrackITS> mTracks;
  std::vector<MCCompLabel> mTrackLabels;

#if TRACKINGITSU_GPU_MODE
  GPU::PrimaryVertexContext mGPUContext;
  GPU::UniquePointer<GPU::PrimaryVertexContext> mGPUContextDevicePointer;
  std::array<GPU::Vector<int>, Constants::ITS::CellsPerRoad> mTempTableArray;
  std::array<GPU::Vector<Tracklet>, Constants::ITS::CellsPerRoad> mTempTrackletArray;
  std::array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad - 1> mTempCellArray;
#else
  std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
             Constants::ITS::TrackletsPerRoad>
    mIndexTables;
  std::array<std::vector<Tracklet>, Constants::ITS::TrackletsPerRoad> mTracklets;
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad> mTrackletsLookupTable;
#endif
};

inline const float3& PrimaryVertexContext::getPrimaryVertex() const { return mPrimaryVertex; }

inline std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& PrimaryVertexContext::getClusters()
{
  return mClusters;
}

inline std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getCells() { return mCells; }

inline std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1>& PrimaryVertexContext::getCellsLookupTable()
{
  return mCellsLookupTable;
}

inline std::array<std::vector<std::vector<int>>, Constants::ITS::CellsPerRoad - 1>&
  PrimaryVertexContext::getCellsNeighbours()
{
  return mCellsNeighbours;
}

inline std::vector<Road>& PrimaryVertexContext::getRoads() { return mRoads; }

inline std::vector<TrackITS>& PrimaryVertexContext::getTracks() { return mTracks; }

inline std::vector<MCCompLabel>& PrimaryVertexContext::getTrackLabels() { return mTrackLabels; }

inline bool PrimaryVertexContext::isClusterUsed(int layer, int clusterId) const
{
  return mUsedClusters[layer][clusterId];
}

inline void PrimaryVertexContext::markUsedCluster(int layer, int clusterId) { mUsedClusters[layer][clusterId] = true; }

#if TRACKINGITSU_GPU_MODE
inline GPU::PrimaryVertexContext& PrimaryVertexContext::getDeviceContext()
{
  return *mGPUContextDevicePointer;
}

inline GPU::Array<GPU::Vector<Cluster>, Constants::ITS::LayersNumber>& PrimaryVertexContext::getDeviceClusters()
{
  return mGPUContext.getClusters();
}

inline GPU::Array<GPU::Vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& PrimaryVertexContext::getDeviceTracklets()
{
  return mGPUContext.getTracklets();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getDeviceTrackletsLookupTable()
{
  return mGPUContext.getTrackletsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>&
  PrimaryVertexContext::getDeviceTrackletsPerClustersTable()
{
  return mGPUContext.getTrackletsPerClusterTable();
}

inline GPU::Array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getDeviceCells()
{
  return mGPUContext.getCells();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad - 1>& PrimaryVertexContext::getDeviceCellsLookupTable()
{
  return mGPUContext.getCellsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad - 1>&
  PrimaryVertexContext::getDeviceCellsPerTrackletTable()
{
  return mGPUContext.getCellsPerTrackletTable();
}

inline std::array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getTempTableArray()
{
  return mTempTableArray;
}

inline std::array<GPU::Vector<Tracklet>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getTempTrackletArray()
{
  return mTempTrackletArray;
}

inline std::array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad - 1>& PrimaryVertexContext::getTempCellArray()
{
  return mTempCellArray;
}

inline void PrimaryVertexContext::updateDeviceContext()
{
  mGPUContextDevicePointer = GPU::UniquePointer<GPU::PrimaryVertexContext>{ mGPUContext };
}
#else
inline std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
                  Constants::ITS::TrackletsPerRoad>&
  PrimaryVertexContext::getIndexTables()
{
  return mIndexTables;
}

inline std::array<std::vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& PrimaryVertexContext::getTracklets()
{
  return mTracklets;
}

inline std::array<std::vector<int>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getTrackletsLookupTable()
{
  return mTrackletsLookupTable;
}
#endif
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXT_H_ */

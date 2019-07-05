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
/// \file DeviceStoreNV.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_DEVICESTORENV_H_
#define TRACKINGITSU_INCLUDE_DEVICESTORENV_H_

#include "ITStracking/Cell.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"
#include "ITStrackingCUDA/Array.h"
#include "ITStrackingCUDA/UniquePointer.h"
#include "ITStrackingCUDA/Vector.h"

namespace o2
{
namespace its
{
namespace GPU
{

class DeviceStoreNV final
{
 public:
  DeviceStoreNV();

  UniquePointer<DeviceStoreNV> initialise(const float3&,
                                          const std::array<std::vector<Cluster>, constants::its::LayersNumber>&,
                                          const std::array<std::vector<Tracklet>, constants::its::TrackletsPerRoad>&,
                                          const std::array<std::vector<Cell>, constants::its::CellsPerRoad>&,
                                          const std::array<std::vector<int>, constants::its::CellsPerRoad - 1>&);
  GPU_DEVICE const float3& getPrimaryVertex();
  GPU_HOST_DEVICE Array<Vector<Cluster>, constants::its::LayersNumber>& getClusters();
  GPU_DEVICE Array<Array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                   constants::its::TrackletsPerRoad>&
    getIndexTables();
  GPU_HOST_DEVICE Array<Vector<Tracklet>, constants::its::TrackletsPerRoad>& getTracklets();
  GPU_HOST_DEVICE Array<Vector<int>, constants::its::CellsPerRoad>& getTrackletsLookupTable();
  GPU_HOST_DEVICE Array<Vector<int>, constants::its::CellsPerRoad>& getTrackletsPerClusterTable();
  GPU_HOST_DEVICE Array<Vector<Cell>, constants::its::CellsPerRoad>& getCells();
  GPU_HOST_DEVICE Array<Vector<int>, constants::its::CellsPerRoad - 1>& getCellsLookupTable();
  GPU_HOST_DEVICE Array<Vector<int>, constants::its::CellsPerRoad - 1>& getCellsPerTrackletTable();
  Array<Vector<int>, constants::its::CellsPerRoad>& getTempTableArray();

 private:
  UniquePointer<float3> mPrimaryVertex;
  Array<Vector<Cluster>, constants::its::LayersNumber> mClusters;
  Array<Array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>, constants::its::TrackletsPerRoad>
    mIndexTables;
  Array<Vector<Tracklet>, constants::its::TrackletsPerRoad> mTracklets;
  Array<Vector<int>, constants::its::CellsPerRoad> mTrackletsLookupTable;
  Array<Vector<int>, constants::its::CellsPerRoad> mTrackletsPerClusterTable;
  Array<Vector<Cell>, constants::its::CellsPerRoad> mCells;
  Array<Vector<int>, constants::its::CellsPerRoad - 1> mCellsLookupTable;
  Array<Vector<int>, constants::its::CellsPerRoad - 1> mCellsPerTrackletTable;
};

GPU_DEVICE inline const float3& DeviceStoreNV::getPrimaryVertex() { return *mPrimaryVertex; }

GPU_HOST_DEVICE inline Array<Vector<Cluster>, constants::its::LayersNumber>& DeviceStoreNV::getClusters()
{
  return mClusters;
}

GPU_DEVICE inline Array<Array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                        constants::its::TrackletsPerRoad>&
  DeviceStoreNV::getIndexTables()
{
  return mIndexTables;
}

GPU_DEVICE inline Array<Vector<Tracklet>, constants::its::TrackletsPerRoad>& DeviceStoreNV::getTracklets()
{
  return mTracklets;
}

GPU_DEVICE inline Array<Vector<int>, constants::its::CellsPerRoad>& DeviceStoreNV::getTrackletsLookupTable()
{
  return mTrackletsLookupTable;
}

GPU_DEVICE inline Array<Vector<int>, constants::its::CellsPerRoad>& DeviceStoreNV::getTrackletsPerClusterTable()
{
  return mTrackletsPerClusterTable;
}

GPU_HOST_DEVICE inline Array<Vector<Cell>, constants::its::CellsPerRoad>& DeviceStoreNV::getCells()
{
  return mCells;
}

GPU_HOST_DEVICE inline Array<Vector<int>, constants::its::CellsPerRoad - 1>& DeviceStoreNV::getCellsLookupTable()
{
  return mCellsLookupTable;
}

GPU_HOST_DEVICE inline Array<Vector<int>, constants::its::CellsPerRoad - 1>&
  DeviceStoreNV::getCellsPerTrackletTable()
{
  return mCellsPerTrackletTable;
}
}
}
}

#endif

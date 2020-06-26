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

#ifndef TRACKINGEC0__INCLUDE_DEVICESTORENV_H_
#define TRACKINGEC0__INCLUDE_DEVICESTORENV_H_

#include "EC0tracking/Cell.h"
#include "EC0tracking/Configuration.h"
#include "EC0tracking/Cluster.h"
#include "EC0tracking/Constants.h"
#include "EC0tracking/Definitions.h"
#include "EC0tracking/Tracklet.h"
#include "EC0trackingCUDA/Array.h"
#include "EC0trackingCUDA/UniquePointer.h"
#include "EC0trackingCUDA/Vector.h"

namespace o2
{
namespace ecl
{
namespace GPU
{

class DeviceStoreNV final
{
 public:
  DeviceStoreNV();

  UniquePointer<DeviceStoreNV> initialise(const float3&,
                                          const std::array<std::vector<Cluster>, constants::ecl::LayersNumber>&,
                                          const std::array<std::vector<Tracklet>, constants::ecl::TrackletsPerRoad>&,
                                          const std::array<std::vector<Cell>, constants::ecl::CellsPerRoad>&,
                                          const std::array<std::vector<int>, constants::ecl::CellsPerRoad - 1>&);
  GPU_DEVICE const float3& getPrimaryVertex();
  GPU_HOST_DEVICE Array<Vector<Cluster>, constants::ecl::LayersNumber>& getClusters();
  GPU_DEVICE Array<Array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                   constants::ecl::TrackletsPerRoad>&
    getIndexTables();
  GPU_HOST_DEVICE Array<Vector<Tracklet>, constants::ecl::TrackletsPerRoad>& getTracklets();
  GPU_HOST_DEVICE Array<Vector<int>, constants::ecl::CellsPerRoad>& getTrackletsLookupTable();
  GPU_HOST_DEVICE Array<Vector<int>, constants::ecl::CellsPerRoad>& getTrackletsPerClusterTable();
  GPU_HOST_DEVICE Array<Vector<Cell>, constants::ecl::CellsPerRoad>& getCells();
  GPU_HOST_DEVICE Array<Vector<int>, constants::ecl::CellsPerRoad - 1>& getCellsLookupTable();
  GPU_HOST_DEVICE Array<Vector<int>, constants::ecl::CellsPerRoad - 1>& getCellsPerTrackletTable();
  Array<Vector<int>, constants::ecl::CellsPerRoad>& getTempTableArray();

 private:
  UniquePointer<float3> mPrimaryVertex;
  Array<Vector<Cluster>, constants::ecl::LayersNumber> mClusters;
  Array<Array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>, constants::ecl::TrackletsPerRoad>
    mIndexTables;
  Array<Vector<Tracklet>, constants::ecl::TrackletsPerRoad> mTracklets;
  Array<Vector<int>, constants::ecl::CellsPerRoad> mTrackletsLookupTable;
  Array<Vector<int>, constants::ecl::CellsPerRoad> mTrackletsPerClusterTable;
  Array<Vector<Cell>, constants::ecl::CellsPerRoad> mCells;
  Array<Vector<int>, constants::ecl::CellsPerRoad - 1> mCellsLookupTable;
  Array<Vector<int>, constants::ecl::CellsPerRoad - 1> mCellsPerTrackletTable;
};

GPU_DEVICE inline const float3& DeviceStoreNV::getPrimaryVertex() { return *mPrimaryVertex; }

GPU_HOST_DEVICE inline Array<Vector<Cluster>, constants::ecl::LayersNumber>& DeviceStoreNV::getClusters()
{
  return mClusters;
}

GPU_DEVICE inline Array<Array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
                        constants::ecl::TrackletsPerRoad>&
  DeviceStoreNV::getIndexTables()
{
  return mIndexTables;
}

GPU_DEVICE inline Array<Vector<Tracklet>, constants::ecl::TrackletsPerRoad>& DeviceStoreNV::getTracklets()
{
  return mTracklets;
}

GPU_DEVICE inline Array<Vector<int>, constants::ecl::CellsPerRoad>& DeviceStoreNV::getTrackletsLookupTable()
{
  return mTrackletsLookupTable;
}

GPU_DEVICE inline Array<Vector<int>, constants::ecl::CellsPerRoad>& DeviceStoreNV::getTrackletsPerClusterTable()
{
  return mTrackletsPerClusterTable;
}

GPU_HOST_DEVICE inline Array<Vector<Cell>, constants::ecl::CellsPerRoad>& DeviceStoreNV::getCells()
{
  return mCells;
}

GPU_HOST_DEVICE inline Array<Vector<int>, constants::ecl::CellsPerRoad - 1>& DeviceStoreNV::getCellsLookupTable()
{
  return mCellsLookupTable;
}

GPU_HOST_DEVICE inline Array<Vector<int>, constants::ecl::CellsPerRoad - 1>&
  DeviceStoreNV::getCellsPerTrackletTable()
{
  return mCellsPerTrackletTable;
}
} // namespace GPU
} // namespace ecl
} // namespace o2

#endif

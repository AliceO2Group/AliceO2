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
namespace ITS
{
namespace GPU
{

class DeviceStoreNV final
{
 public:
  DeviceStoreNV();

  UniquePointer<DeviceStoreNV> initialise(const float3&,
                                                 const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>&,
                                                 const std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad>&,
                                                 const std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1>&,
                                                 const MemoryParameters&);
  GPU_DEVICE const float3& getPrimaryVertex();
  GPU_HOST_DEVICE Array<Vector<Cluster>, Constants::ITS::LayersNumber>& getClusters();
  GPU_DEVICE Array<Array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
                   Constants::ITS::TrackletsPerRoad>&
    getIndexTables();
  GPU_HOST_DEVICE Array<Vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& getTracklets();
  GPU_HOST_DEVICE Array<Vector<int>, Constants::ITS::CellsPerRoad>& getTrackletsLookupTable();
  GPU_HOST_DEVICE Array<Vector<int>, Constants::ITS::CellsPerRoad>& getTrackletsPerClusterTable();
  GPU_HOST_DEVICE Array<Vector<Cell>, Constants::ITS::CellsPerRoad>& getCells();
  GPU_HOST_DEVICE Array<Vector<int>, Constants::ITS::CellsPerRoad - 1>& getCellsLookupTable();
  GPU_HOST_DEVICE Array<Vector<int>, Constants::ITS::CellsPerRoad - 1>& getCellsPerTrackletTable();
  Array<Vector<int>, Constants::ITS::CellsPerRoad>& getTempTableArray();

 private:
  UniquePointer<float3> mPrimaryVertex;
  Array<Vector<Cluster>, Constants::ITS::LayersNumber> mClusters;
  Array<Array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>, Constants::ITS::TrackletsPerRoad>
    mIndexTables;
  Array<Vector<Tracklet>, Constants::ITS::TrackletsPerRoad> mTracklets;
  Array<Vector<int>, Constants::ITS::CellsPerRoad> mTrackletsLookupTable;
  Array<Vector<int>, Constants::ITS::CellsPerRoad> mTrackletsPerClusterTable;
  Array<Vector<Cell>, Constants::ITS::CellsPerRoad> mCells;
  Array<Vector<int>, Constants::ITS::CellsPerRoad - 1> mCellsLookupTable;
  Array<Vector<int>, Constants::ITS::CellsPerRoad - 1> mCellsPerTrackletTable;
};

GPU_DEVICE inline const float3& DeviceStoreNV::getPrimaryVertex() { return *mPrimaryVertex; }

GPU_HOST_DEVICE inline Array<Vector<Cluster>, Constants::ITS::LayersNumber>& DeviceStoreNV::getClusters()
{
  return mClusters;
}

GPU_DEVICE inline Array<Array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
                        Constants::ITS::TrackletsPerRoad>&
  DeviceStoreNV::getIndexTables()
{
  return mIndexTables;
}

GPU_DEVICE inline Array<Vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& DeviceStoreNV::getTracklets()
{
  return mTracklets;
}

GPU_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad>& DeviceStoreNV::getTrackletsLookupTable()
{
  return mTrackletsLookupTable;
}

GPU_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad>& DeviceStoreNV::getTrackletsPerClusterTable()
{
  return mTrackletsPerClusterTable;
}

GPU_HOST_DEVICE inline Array<Vector<Cell>, Constants::ITS::CellsPerRoad>& DeviceStoreNV::getCells()
{
  return mCells;
}

GPU_HOST_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad - 1>& DeviceStoreNV::getCellsLookupTable()
{
  return mCellsLookupTable;
}

GPU_HOST_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad - 1>&
  DeviceStoreNV::getCellsPerTrackletTable()
{
  return mCellsPerTrackletTable;
}
}
}
}

#endif
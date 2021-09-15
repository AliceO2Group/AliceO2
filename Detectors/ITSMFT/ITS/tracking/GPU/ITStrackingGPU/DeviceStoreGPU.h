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
/// \file DeviceStoreNV.h
/// \brief
///

#ifndef ITSTRACKINGGPU_DEVICESTOREGPU_H_
#define ITSTRACKINGGPU_DEVICESTOREGPU_H_

#ifndef GPUCA_GPUCODE_GENRTC
#include <cub/cub.cuh>
#include <cstdint>
#endif

#include "ITStracking/Cell.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Tracklet.h"

#include "Array.h"
#include "UniquePointer.h"
#include "Vector.h"

namespace o2
{
namespace its
{

namespace gpu
{

class DeviceStoreNV final
{
 public:
  DeviceStoreNV();

  UniquePointer<DeviceStoreNV> initialise(const float3&,
                                          const std::array<std::vector<Cluster>, constants::its2::LayersNumber>&,
                                          const std::array<std::vector<Tracklet>, constants::its2::TrackletsPerRoad>&,
                                          const std::array<std::vector<Cell>, constants::its2::CellsPerRoad>&,
                                          const std::array<std::vector<int>, constants::its2::CellsPerRoad - 1>&,
                                          const std::array<float, constants::its2::LayersNumber>&,
                                          const std::array<float, constants::its2::LayersNumber>&);
  GPUd() const float3& getPrimaryVertex() { return *mPrimaryVertex; };
  GPUhd() Array<Vector<Cluster>, constants::its2::LayersNumber>& getClusters();
  GPUd() Array<Array<int, constants::its2::ZBins * constants::its2::PhiBins + 1>,
               constants::its2::TrackletsPerRoad>& getIndexTables() { return mIndexTables; };
  GPUhd() Array<Vector<Tracklet>, constants::its2::TrackletsPerRoad>& getTracklets();
  GPUhd() Array<Vector<int>, constants::its2::CellsPerRoad>& getTrackletsLookupTable();
  GPUhd() Array<Vector<int>, constants::its2::CellsPerRoad>& getTrackletsPerClusterTable();
  GPUhd() Array<Vector<Cell>, constants::its2::CellsPerRoad>& getCells();
  GPUhd() Array<Vector<int>, constants::its2::CellsPerRoad - 1>& getCellsLookupTable();
  GPUhd() Array<Vector<int>, constants::its2::CellsPerRoad - 1>& getCellsPerTrackletTable();
  Array<Vector<int>, constants::its2::CellsPerRoad>& getTempTableArray();

  GPUhd() float getRmin(int layer);
  GPUhd() float getRmax(int layer);

 private:
  UniquePointer<float3> mPrimaryVertex;
  Array<Vector<Cluster>, constants::its2::LayersNumber> mClusters;
  Array<float, constants::its2::LayersNumber> mRmin;
  Array<float, constants::its2::LayersNumber> mRmax;
  Array<Array<int, constants::its2::ZBins * constants::its2::PhiBins + 1>, constants::its2::TrackletsPerRoad>
    mIndexTables;
  Array<Vector<Tracklet>, constants::its2::TrackletsPerRoad> mTracklets;
  Array<Vector<int>, constants::its2::CellsPerRoad> mTrackletsLookupTable;
  Array<Vector<int>, constants::its2::CellsPerRoad> mTrackletsPerClusterTable;
  Array<Vector<Cell>, constants::its2::CellsPerRoad> mCells;
  Array<Vector<int>, constants::its2::CellsPerRoad - 1> mCellsLookupTable;
  Array<Vector<int>, constants::its2::CellsPerRoad - 1> mCellsPerTrackletTable;
};

GPUhd() Array<Vector<Cluster>, constants::its2::LayersNumber>& DeviceStoreNV::getClusters()
{
  return mClusters;
}

GPUd() Array<Vector<Tracklet>, constants::its2::TrackletsPerRoad>& DeviceStoreNV::getTracklets()
{
  return mTracklets;
}

GPUd() Array<Vector<int>, constants::its2::CellsPerRoad>& DeviceStoreNV::getTrackletsLookupTable()
{
  return mTrackletsLookupTable;
}

GPUd() Array<Vector<int>, constants::its2::CellsPerRoad>& DeviceStoreNV::getTrackletsPerClusterTable()
{
  return mTrackletsPerClusterTable;
}

GPUhd() Array<Vector<Cell>, constants::its2::CellsPerRoad>& DeviceStoreNV::getCells()
{
  return mCells;
}

GPUhd() Array<Vector<int>, constants::its2::CellsPerRoad - 1>& DeviceStoreNV::getCellsLookupTable()
{
  return mCellsLookupTable;
}

GPUhd() Array<Vector<int>, constants::its2::CellsPerRoad - 1>& DeviceStoreNV::getCellsPerTrackletTable()
{
  return mCellsPerTrackletTable;
}

GPUhd() float DeviceStoreNV::getRmin(int layer)
{
  return mRmin[layer];
}

GPUhd() float DeviceStoreNV::getRmax(int layer)
{
  return mRmax[layer];
}

} // namespace gpu
} // namespace its
} // namespace o2
#endif

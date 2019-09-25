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
/// \file PrimaryVertexContextNVNV.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXTNVNV_H_
#define TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXTNVNV_H_

#include <array>

#include "ITStracking/Configuration.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/PrimaryVertexContext.h"
#include "ITStracking/Road.h"
#include "ITStracking/Tracklet.h"

#include "ITStrackingCUDA/DeviceStoreNV.h"
#include "ITStrackingCUDA/UniquePointer.h"

namespace o2
{
namespace its
{

class PrimaryVertexContextNV final : public PrimaryVertexContext
{
 public:
  PrimaryVertexContextNV() = default;
  virtual ~PrimaryVertexContextNV() = default;

  virtual void initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, constants::its::LayersNumber>& cl,
                          const std::array<float, 3>& pv, const int iteration);

  GPU::DeviceStoreNV& getDeviceContext();
  GPU::Array<GPU::Vector<Cluster>, constants::its::LayersNumber>& getDeviceClusters();
  GPU::Array<GPU::Vector<Tracklet>, constants::its::TrackletsPerRoad>& getDeviceTracklets();
  GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad>& getDeviceTrackletsLookupTable();
  GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad>& getDeviceTrackletsPerClustersTable();
  GPU::Array<GPU::Vector<Cell>, constants::its::CellsPerRoad>& getDeviceCells();
  GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad - 1>& getDeviceCellsLookupTable();
  GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad - 1>& getDeviceCellsPerTrackletTable();
  std::array<GPU::Vector<int>, constants::its::CellsPerRoad>& getTempTableArray();
  std::array<GPU::Vector<Tracklet>, constants::its::CellsPerRoad>& getTempTrackletArray();
  std::array<GPU::Vector<Cell>, constants::its::CellsPerRoad - 1>& getTempCellArray();
  void updateDeviceContext();

 private:
  GPU::DeviceStoreNV mGPUContext;
  GPU::UniquePointer<GPU::DeviceStoreNV> mGPUContextDevicePointer;
  std::array<GPU::Vector<int>, constants::its::CellsPerRoad> mTempTableArray;
  std::array<GPU::Vector<Tracklet>, constants::its::CellsPerRoad> mTempTrackletArray;
  std::array<GPU::Vector<Cell>, constants::its::CellsPerRoad - 1> mTempCellArray;
};

inline GPU::DeviceStoreNV& PrimaryVertexContextNV::getDeviceContext()
{
  return *mGPUContextDevicePointer;
}

inline GPU::Array<GPU::Vector<Cluster>, constants::its::LayersNumber>& PrimaryVertexContextNV::getDeviceClusters()
{
  return mGPUContext.getClusters();
}

inline GPU::Array<GPU::Vector<Tracklet>, constants::its::TrackletsPerRoad>& PrimaryVertexContextNV::getDeviceTracklets()
{
  return mGPUContext.getTracklets();
}

inline GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad>& PrimaryVertexContextNV::getDeviceTrackletsLookupTable()
{
  return mGPUContext.getTrackletsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad>&
  PrimaryVertexContextNV::getDeviceTrackletsPerClustersTable()
{
  return mGPUContext.getTrackletsPerClusterTable();
}

inline GPU::Array<GPU::Vector<Cell>, constants::its::CellsPerRoad>& PrimaryVertexContextNV::getDeviceCells()
{
  return mGPUContext.getCells();
}

inline GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad - 1>& PrimaryVertexContextNV::getDeviceCellsLookupTable()
{
  return mGPUContext.getCellsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, constants::its::CellsPerRoad - 1>&
  PrimaryVertexContextNV::getDeviceCellsPerTrackletTable()
{
  return mGPUContext.getCellsPerTrackletTable();
}

inline std::array<GPU::Vector<int>, constants::its::CellsPerRoad>& PrimaryVertexContextNV::getTempTableArray()
{
  return mTempTableArray;
}

inline std::array<GPU::Vector<Tracklet>, constants::its::CellsPerRoad>& PrimaryVertexContextNV::getTempTrackletArray()
{
  return mTempTrackletArray;
}

inline std::array<GPU::Vector<Cell>, constants::its::CellsPerRoad - 1>& PrimaryVertexContextNV::getTempCellArray()
{
  return mTempCellArray;
}

inline void PrimaryVertexContextNV::updateDeviceContext()
{
  mGPUContextDevicePointer = GPU::UniquePointer<GPU::DeviceStoreNV>{mGPUContext};
}

inline void PrimaryVertexContextNV::initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, constants::its::LayersNumber>& cl,
                                               const std::array<float, 3>& pv, const int iteration)
{
  this->PrimaryVertexContext::initialise(memParam, cl, pv, iteration);
  mGPUContextDevicePointer = mGPUContext.initialise(mPrimaryVertex, mClusters, mTracklets, mCells, mCellsLookupTable);
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXTNV_H_ */

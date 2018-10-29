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
namespace ITS
{

class PrimaryVertexContextNV final : public PrimaryVertexContext
{
 public:
  PrimaryVertexContextNV() = default;
  virtual ~PrimaryVertexContextNV() = default;

  virtual void initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& cl,
                          const float3& pv, const int iteration);

  GPU::DeviceStoreNV& getDeviceContext();
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

 private:
  GPU::DeviceStoreNV mGPUContext;
  GPU::UniquePointer<GPU::DeviceStoreNV> mGPUContextDevicePointer;
  std::array<GPU::Vector<int>, Constants::ITS::CellsPerRoad> mTempTableArray;
  std::array<GPU::Vector<Tracklet>, Constants::ITS::CellsPerRoad> mTempTrackletArray;
  std::array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad - 1> mTempCellArray;
};

inline GPU::DeviceStoreNV& PrimaryVertexContextNV::getDeviceContext()
{
  return *mGPUContextDevicePointer;
}

inline GPU::Array<GPU::Vector<Cluster>, Constants::ITS::LayersNumber>& PrimaryVertexContextNV::getDeviceClusters()
{
  return mGPUContext.getClusters();
}

inline GPU::Array<GPU::Vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& PrimaryVertexContextNV::getDeviceTracklets()
{
  return mGPUContext.getTracklets();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>& PrimaryVertexContextNV::getDeviceTrackletsLookupTable()
{
  return mGPUContext.getTrackletsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>&
  PrimaryVertexContextNV::getDeviceTrackletsPerClustersTable()
{
  return mGPUContext.getTrackletsPerClusterTable();
}

inline GPU::Array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad>& PrimaryVertexContextNV::getDeviceCells()
{
  return mGPUContext.getCells();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad - 1>& PrimaryVertexContextNV::getDeviceCellsLookupTable()
{
  return mGPUContext.getCellsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, Constants::ITS::CellsPerRoad - 1>&
  PrimaryVertexContextNV::getDeviceCellsPerTrackletTable()
{
  return mGPUContext.getCellsPerTrackletTable();
}

inline std::array<GPU::Vector<int>, Constants::ITS::CellsPerRoad>& PrimaryVertexContextNV::getTempTableArray()
{
  return mTempTableArray;
}

inline std::array<GPU::Vector<Tracklet>, Constants::ITS::CellsPerRoad>& PrimaryVertexContextNV::getTempTrackletArray()
{
  return mTempTrackletArray;
}

inline std::array<GPU::Vector<Cell>, Constants::ITS::CellsPerRoad - 1>& PrimaryVertexContextNV::getTempCellArray()
{
  return mTempCellArray;
}

inline void PrimaryVertexContextNV::updateDeviceContext()
{
  mGPUContextDevicePointer = GPU::UniquePointer<GPU::DeviceStoreNV>{ mGPUContext };
}

inline void PrimaryVertexContextNV::initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& cl,
                          const float3& pv, const int iteration)
{
  PrimaryVertexContext::initialise(memParam, cl, pv, iteration);
  mGPUContextDevicePointer = mGPUContext.initialise(mPrimaryVertex, mClusters, mCells, mCellsLookupTable, memParam);
}

} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXTNV_H_ */

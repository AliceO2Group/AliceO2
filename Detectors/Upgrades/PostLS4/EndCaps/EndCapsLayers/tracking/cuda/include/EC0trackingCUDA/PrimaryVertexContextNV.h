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

#ifndef TRACKINGEC0__INCLUDE_PRIMARYVERTEXCONTEXTNVNV_H_
#define TRACKINGEC0__INCLUDE_PRIMARYVERTEXCONTEXTNVNV_H_

#include <array>

#include "EC0tracking/Configuration.h"
#include "EC0tracking/Constants.h"
#include "EC0tracking/Definitions.h"
#include "EC0tracking/PrimaryVertexContext.h"
#include "EC0tracking/Road.h"
#include "EC0tracking/Tracklet.h"

#include "EC0trackingCUDA/DeviceStoreNV.h"
#include "EC0trackingCUDA/UniquePointer.h"

namespace o2
{
namespace ecl
{

class PrimaryVertexContextNV final : public PrimaryVertexContext
{
 public:
  PrimaryVertexContextNV() = default;
  virtual ~PrimaryVertexContextNV() = default;

  virtual void initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, constants::ecl::LayersNumber>& cl,
                          const std::array<float, 3>& pv, const int iteration);

  GPU::DeviceStoreNV& getDeviceContext();
  GPU::Array<GPU::Vector<Cluster>, constants::ecl::LayersNumber>& getDeviceClusters();
  GPU::Array<GPU::Vector<Tracklet>, constants::ecl::TrackletsPerRoad>& getDeviceTracklets();
  GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad>& getDeviceTrackletsLookupTable();
  GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad>& getDeviceTrackletsPerClustersTable();
  GPU::Array<GPU::Vector<Cell>, constants::ecl::CellsPerRoad>& getDeviceCells();
  GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad - 1>& getDeviceCellsLookupTable();
  GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad - 1>& getDeviceCellsPerTrackletTable();
  std::array<GPU::Vector<int>, constants::ecl::CellsPerRoad>& getTempTableArray();
  std::array<GPU::Vector<Tracklet>, constants::ecl::CellsPerRoad>& getTempTrackletArray();
  std::array<GPU::Vector<Cell>, constants::ecl::CellsPerRoad - 1>& getTempCellArray();
  void updateDeviceContext();

 private:
  GPU::DeviceStoreNV mGPUContext;
  GPU::UniquePointer<GPU::DeviceStoreNV> mGPUContextDevicePointer;
  std::array<GPU::Vector<int>, constants::ecl::CellsPerRoad> mTempTableArray;
  std::array<GPU::Vector<Tracklet>, constants::ecl::CellsPerRoad> mTempTrackletArray;
  std::array<GPU::Vector<Cell>, constants::ecl::CellsPerRoad - 1> mTempCellArray;
};

inline GPU::DeviceStoreNV& PrimaryVertexContextNV::getDeviceContext()
{
  return *mGPUContextDevicePointer;
}

inline GPU::Array<GPU::Vector<Cluster>, constants::ecl::LayersNumber>& PrimaryVertexContextNV::getDeviceClusters()
{
  return mGPUContext.getClusters();
}

inline GPU::Array<GPU::Vector<Tracklet>, constants::ecl::TrackletsPerRoad>& PrimaryVertexContextNV::getDeviceTracklets()
{
  return mGPUContext.getTracklets();
}

inline GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad>& PrimaryVertexContextNV::getDeviceTrackletsLookupTable()
{
  return mGPUContext.getTrackletsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad>&
  PrimaryVertexContextNV::getDeviceTrackletsPerClustersTable()
{
  return mGPUContext.getTrackletsPerClusterTable();
}

inline GPU::Array<GPU::Vector<Cell>, constants::ecl::CellsPerRoad>& PrimaryVertexContextNV::getDeviceCells()
{
  return mGPUContext.getCells();
}

inline GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad - 1>& PrimaryVertexContextNV::getDeviceCellsLookupTable()
{
  return mGPUContext.getCellsLookupTable();
}

inline GPU::Array<GPU::Vector<int>, constants::ecl::CellsPerRoad - 1>&
  PrimaryVertexContextNV::getDeviceCellsPerTrackletTable()
{
  return mGPUContext.getCellsPerTrackletTable();
}

inline std::array<GPU::Vector<int>, constants::ecl::CellsPerRoad>& PrimaryVertexContextNV::getTempTableArray()
{
  return mTempTableArray;
}

inline std::array<GPU::Vector<Tracklet>, constants::ecl::CellsPerRoad>& PrimaryVertexContextNV::getTempTrackletArray()
{
  return mTempTrackletArray;
}

inline std::array<GPU::Vector<Cell>, constants::ecl::CellsPerRoad - 1>& PrimaryVertexContextNV::getTempCellArray()
{
  return mTempCellArray;
}

inline void PrimaryVertexContextNV::updateDeviceContext()
{
  mGPUContextDevicePointer = GPU::UniquePointer<GPU::DeviceStoreNV>{mGPUContext};
}

inline void PrimaryVertexContextNV::initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, constants::ecl::LayersNumber>& cl,
                                               const std::array<float, 3>& pv, const int iteration)
{
  this->PrimaryVertexContext::initialise(memParam, cl, pv, iteration);
  mGPUContextDevicePointer = mGPUContext.initialise(mPrimaryVertex, mClusters, mTracklets, mCells, mCellsLookupTable);
}

} // namespace ecl
} // namespace o2

#endif /* TRACKINGEC0__INCLUDE_PRIMARYVERTEXCONTEXTNV_H_ */

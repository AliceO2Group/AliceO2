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
  ~PrimaryVertexContextNV() override;

  void initialise(const MemoryParameters& memParam, const TrackingParameters& trkParam,
                  const std::vector<std::vector<Cluster>>& cl, const std::array<float, 3>& pv, const int iteration) override;

  gpu::DeviceStoreNV& getDeviceContext();
  gpu::Array<gpu::Vector<Cluster>, constants::its2::LayersNumber>& getDeviceClusters();
  gpu::Array<gpu::Vector<Tracklet>, constants::its2::TrackletsPerRoad>& getDeviceTracklets();
  gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad>& getDeviceTrackletsLookupTable();
  gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad>& getDeviceTrackletsPerClustersTable();
  gpu::Array<gpu::Vector<Cell>, constants::its2::CellsPerRoad>& getDeviceCells();
  gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad - 1>& getDeviceCellsLookupTable();
  gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad - 1>& getDeviceCellsPerTrackletTable();
  std::array<gpu::Vector<int>, constants::its2::CellsPerRoad>& getTempTableArray();
  std::array<gpu::Vector<Tracklet>, constants::its2::CellsPerRoad>& getTempTrackletArray();
  std::array<gpu::Vector<Cell>, constants::its2::CellsPerRoad - 1>& getTempCellArray();
  void updateDeviceContext();

 private:
  gpu::DeviceStoreNV mGPUContext;
  gpu::UniquePointer<gpu::DeviceStoreNV> mGPUContextDevicePointer;
  std::array<gpu::Vector<int>, constants::its2::CellsPerRoad> mTempTableArray;
  std::array<gpu::Vector<Tracklet>, constants::its2::CellsPerRoad> mTempTrackletArray;
  std::array<gpu::Vector<Cell>, constants::its2::CellsPerRoad - 1> mTempCellArray;
};

inline PrimaryVertexContextNV::~PrimaryVertexContextNV() = default;

inline gpu::DeviceStoreNV& PrimaryVertexContextNV::getDeviceContext()
{
  return *mGPUContextDevicePointer;
}

inline gpu::Array<gpu::Vector<Cluster>, constants::its2::LayersNumber>& PrimaryVertexContextNV::getDeviceClusters()
{
  return mGPUContext.getClusters();
}

inline gpu::Array<gpu::Vector<Tracklet>, constants::its2::TrackletsPerRoad>& PrimaryVertexContextNV::getDeviceTracklets()
{
  return mGPUContext.getTracklets();
}

inline gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad>& PrimaryVertexContextNV::getDeviceTrackletsLookupTable()
{
  return mGPUContext.getTrackletsLookupTable();
}

inline gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad>&
  PrimaryVertexContextNV::getDeviceTrackletsPerClustersTable()
{
  return mGPUContext.getTrackletsPerClusterTable();
}

inline gpu::Array<gpu::Vector<Cell>, constants::its2::CellsPerRoad>& PrimaryVertexContextNV::getDeviceCells()
{
  return mGPUContext.getCells();
}

inline gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad - 1>& PrimaryVertexContextNV::getDeviceCellsLookupTable()
{
  return mGPUContext.getCellsLookupTable();
}

inline gpu::Array<gpu::Vector<int>, constants::its2::CellsPerRoad - 1>&
  PrimaryVertexContextNV::getDeviceCellsPerTrackletTable()
{
  return mGPUContext.getCellsPerTrackletTable();
}

inline std::array<gpu::Vector<int>, constants::its2::CellsPerRoad>& PrimaryVertexContextNV::getTempTableArray()
{
  return mTempTableArray;
}

inline std::array<gpu::Vector<Tracklet>, constants::its2::CellsPerRoad>& PrimaryVertexContextNV::getTempTrackletArray()
{
  return mTempTrackletArray;
}

inline std::array<gpu::Vector<Cell>, constants::its2::CellsPerRoad - 1>& PrimaryVertexContextNV::getTempCellArray()
{
  return mTempCellArray;
}

inline void PrimaryVertexContextNV::updateDeviceContext()
{
  mGPUContextDevicePointer = gpu::UniquePointer<gpu::DeviceStoreNV>{mGPUContext};
}

inline void PrimaryVertexContextNV::initialise(const MemoryParameters& memParam, const TrackingParameters& trkParam,
                                               const std::vector<std::vector<Cluster>>& cl, const std::array<float, 3>& pv, const int iteration)
{
  ///TODO: to be re-enabled in the future
  // this->PrimaryVertexContext::initialise(memParam, cl, pv, iteration);
  // mGPUContextDevicePointer = mGPUContext.initialise(mPrimaryVertex, mClusters, mTracklets, mCells, mCellsLookupTable, mMinR, mMaxR);
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXTNV_H_ */

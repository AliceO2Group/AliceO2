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

#include "ITSReconstruction/CA/Cell.h"
#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Tracklet.h"
#include "ITSReconstruction/CA/gpu/Array.h"
#include "ITSReconstruction/CA/gpu/UniquePointer.h"
#include "ITSReconstruction/CA/gpu/Vector.h"

namespace o2
{
namespace ITS
{
namespace CA
{
namespace GPU
{

class PrimaryVertexContext
  final
  {
    public:
      PrimaryVertexContext();

      UniquePointer<PrimaryVertexContext> initialize(const float3&,
          const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>&,
          const std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad>&,
          const std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1>&);
      GPU_DEVICE const float3& getPrimaryVertex();
      GPU_HOST_DEVICE Array<Vector<Cluster>,
          Constants::ITS::LayersNumber>& getClusters();
      GPU_DEVICE Array<Array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
          Constants::ITS::TrackletsPerRoad>& getIndexTables();
      GPU_HOST_DEVICE Array<Vector<Tracklet>,
          Constants::ITS::TrackletsPerRoad>& getTracklets();
      GPU_HOST_DEVICE Array<Vector<int>,
          Constants::ITS::CellsPerRoad>& getTrackletsLookupTable();
      GPU_HOST_DEVICE Array<Vector<int>,
          Constants::ITS::CellsPerRoad>& getTrackletsPerClusterTable();
      GPU_HOST_DEVICE Array<Vector<Cell>,
          Constants::ITS::CellsPerRoad>& getCells();
      GPU_HOST_DEVICE Array<Vector<int>,
          Constants::ITS::CellsPerRoad - 1>& getCellsLookupTable();
      GPU_HOST_DEVICE Array<Vector<int>,
          Constants::ITS::CellsPerRoad - 1>& getCellsPerTrackletTable();
     Array<Vector<int>, Constants::ITS::CellsPerRoad>& getTempTableArray();

    private:
      UniquePointer<float3> mPrimaryVertex;
      Array<Vector<Cluster>, Constants::ITS::LayersNumber> mClusters;
      Array<Array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
          Constants::ITS::TrackletsPerRoad> mIndexTables;
      Array<Vector<Tracklet>, Constants::ITS::TrackletsPerRoad> mTracklets;
      Array<Vector<int>, Constants::ITS::CellsPerRoad> mTrackletsLookupTable;
      Array<Vector<int>, Constants::ITS::CellsPerRoad> mTrackletsPerClusterTable;
      Array<Vector<Cell>, Constants::ITS::CellsPerRoad> mCells;
      Array<Vector<int>, Constants::ITS::CellsPerRoad - 1> mCellsLookupTable;
      Array<Vector<int>, Constants::ITS::CellsPerRoad - 1> mCellsPerTrackletTable;
  };

  GPU_DEVICE inline const float3& PrimaryVertexContext::getPrimaryVertex()
  {
    return *mPrimaryVertex;
  }

  GPU_HOST_DEVICE inline Array<Vector<Cluster>, Constants::ITS::LayersNumber>& PrimaryVertexContext::getClusters()
  {
    return mClusters;
  }

  GPU_DEVICE inline Array<Array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
      Constants::ITS::TrackletsPerRoad>& PrimaryVertexContext::getIndexTables()
  {
    return mIndexTables;
  }

  GPU_DEVICE inline Array<Vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& PrimaryVertexContext::getTracklets()
  {
    return mTracklets;
  }

  GPU_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getTrackletsLookupTable()
  {
    return mTrackletsLookupTable;
  }

  GPU_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getTrackletsPerClusterTable()
  {
    return mTrackletsPerClusterTable;
  }

  GPU_HOST_DEVICE inline Array<Vector<Cell>, Constants::ITS::CellsPerRoad>& PrimaryVertexContext::getCells()
  {
    return mCells;
  }

  GPU_HOST_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad - 1>& PrimaryVertexContext::getCellsLookupTable()
  {
    return mCellsLookupTable;
  }

  GPU_HOST_DEVICE inline Array<Vector<int>, Constants::ITS::CellsPerRoad - 1>& PrimaryVertexContext::getCellsPerTrackletTable()
  {
    return mCellsPerTrackletTable;
  }

}
}
}
}

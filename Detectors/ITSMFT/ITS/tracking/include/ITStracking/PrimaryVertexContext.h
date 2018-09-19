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

#include "ITStracking/Cell.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Road.h"
#include "ITStracking/Tracklet.h"

namespace o2
{
namespace ITS
{

class PrimaryVertexContext
{
 public:
  PrimaryVertexContext();

  virtual ~PrimaryVertexContext() = default;

  PrimaryVertexContext(const PrimaryVertexContext&) = delete;
  PrimaryVertexContext& operator=(const PrimaryVertexContext&) = delete;

  virtual void initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& cl,
                          const std::array<float,3> &pv, const int iteration);
  const float3& getPrimaryVertex() const;
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumber>& getClusters();
  std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad>& getCells();
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1>& getCellsLookupTable();
  std::array<std::vector<std::vector<int>>, Constants::ITS::CellsPerRoad - 1>& getCellsNeighbours();
  std::vector<Road>& getRoads();

  bool isClusterUsed(int layer, int clusterId) const;
  void markUsedCluster(int layer, int clusterId);

  std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
             Constants::ITS::TrackletsPerRoad>&
    getIndexTables();
  std::array<std::vector<Tracklet>, Constants::ITS::TrackletsPerRoad>& getTracklets();
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad>& getTrackletsLookupTable();

 protected:
  float3 mPrimaryVertex;
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumber> mClusters;
  std::array<std::vector<bool>, Constants::ITS::LayersNumber> mUsedClusters;
  std::array<std::vector<Cell>, Constants::ITS::CellsPerRoad> mCells;
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad - 1> mCellsLookupTable;
  std::array<std::vector<std::vector<int>>, Constants::ITS::CellsPerRoad - 1> mCellsNeighbours;
  std::vector<Road> mRoads;

  std::array<std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1>,
             Constants::ITS::TrackletsPerRoad>
    mIndexTables;
  std::array<std::vector<Tracklet>, Constants::ITS::TrackletsPerRoad> mTracklets;
  std::array<std::vector<int>, Constants::ITS::CellsPerRoad> mTrackletsLookupTable;
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

inline bool PrimaryVertexContext::isClusterUsed(int layer, int clusterId) const
{
  return mUsedClusters[layer][clusterId];
}

inline void PrimaryVertexContext::markUsedCluster(int layer, int clusterId) { mUsedClusters[layer][clusterId] = true; }

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
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXT_H_ */

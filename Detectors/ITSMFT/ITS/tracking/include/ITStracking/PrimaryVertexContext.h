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
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/Road.h"
#include "ITStracking/Tracklet.h"

namespace o2
{
namespace its
{

class PrimaryVertexContext
{
 public:
  PrimaryVertexContext() = default;

  virtual ~PrimaryVertexContext() = default;

  PrimaryVertexContext(const PrimaryVertexContext&) = delete;
  PrimaryVertexContext& operator=(const PrimaryVertexContext&) = delete;

  virtual void initialise(const MemoryParameters& memParam, const TrackingParameters& trkParam,
                          const std::vector<std::vector<Cluster>>& cl, const std::array<float, 3>& pv, const int iteration);
  const float3& getPrimaryVertex() const { return mPrimaryVertex; }
  auto& getClusters() { return mClusters; }
  auto& getCells() { return mCells; }
  auto& getCellsLookupTable() { return mCellsLookupTable; }
  auto& getCellsNeighbours() { return mCellsNeighbours; }
  auto& getRoads() { return mRoads; }

  float getMinR(int layer) { return mMinR[layer]; }
  float getMaxR(int layer) { return mMaxR[layer]; }

  bool isClusterUsed(int layer, int clusterId) const { return mUsedClusters[layer][clusterId]; }
  void markUsedCluster(int layer, int clusterId);

  auto& getIndexTables() { return mIndexTables; }
  auto& getTracklets() { return mTracklets; }
  auto& getTrackletsLookupTable() { return mTrackletsLookupTable; }

  void initialiseRoadLabels();
  void setRoadLabel(int i, const unsigned long long& lab, bool fake);
  const unsigned long long& getRoadLabel(int i) const;
  bool isRoadFake(int i) const;

  IndexTableUtils mIndexTableUtils;

 protected:
  float3 mPrimaryVertex;
  std::vector<float> mMinR;
  std::vector<float> mMaxR;
  std::vector<std::vector<Cluster>> mUnsortedClusters;
  std::vector<std::vector<Cluster>> mClusters;
  std::vector<std::vector<bool>> mUsedClusters;
  std::vector<std::vector<Cell>> mCells;
  std::vector<std::vector<int>> mCellsLookupTable;
  std::vector<std::vector<std::vector<int>>> mCellsNeighbours;
  std::vector<Road> mRoads;

  // std::array<std::array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>,
  //            constants::its::TrackletsPerRoad>
  std::vector<std::vector<int>> mIndexTables;
  std::vector<std::vector<Tracklet>> mTracklets;
  std::vector<std::vector<int>> mTrackletsLookupTable;

  std::vector<std::pair<unsigned long long, bool>> mRoadLabels;
};


inline void PrimaryVertexContext::markUsedCluster(int layer, int clusterId) { mUsedClusters[layer][clusterId] = true; }

inline void PrimaryVertexContext::initialiseRoadLabels()
{
  mRoadLabels.clear();
  mRoadLabels.resize(mRoads.size());
}

inline void PrimaryVertexContext::setRoadLabel(int i, const unsigned long long& lab, bool fake)
{
  mRoadLabels[i].first = lab;
  mRoadLabels[i].second = fake;
}

inline const unsigned long long& PrimaryVertexContext::getRoadLabel(int i) const
{
  return mRoadLabels[i].first;
}

inline bool PrimaryVertexContext::isRoadFake(int i) const
{
  return mRoadLabels[i].second;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_PRIMARYVERTEXCONTEXT_H_ */

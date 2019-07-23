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
/// \file ROframe.h
/// \brief The main container for the standalone track finding within a read-out-frame
///

#ifndef O2_MFT_ROFRAME_H_
#define O2_MFT_ROFRAME_H_

#include <array>
#include <vector>
#include <utility>
#include <cassert>
#include <gsl/gsl>
#include <map>

#include "MFTTracking/Cluster.h"
#include "MFTTracking/Constants.h"
#include "MFTTracking/Cell.h"
#include "MFTTracking/TrackCA.h"
#include "MFTTracking/Road.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace mft
{

class TrackCA;
class TrackLTF;

class ROframe final
{
 public:
  ROframe(Int_t ROframeId);
  Int_t getROFrameId() const;
  Int_t getTotalClusters() const;

  void setROFrameId(const Int_t rofid) { mROframeId = rofid; }

  const std::array<std::vector<Cluster>, Constants::mft::LayersNumber>& getClusters() const;
  const std::vector<Cluster>& getClustersInLayer(Int_t layerId) const;
  const MCCompLabel& getClusterLabels(Int_t layerId, const Cluster& cl) const;
  const MCCompLabel& getClusterLabels(Int_t layerId, const Int_t clId) const;
  const std::map<Int_t, std::pair<Int_t, Int_t>>& getClusterBinIndexRange(Int_t layerId) const;
  Int_t getClusterExternalIndex(Int_t layerId, const Int_t clId) const;
  const std::array<std::vector<Cell>, Constants::mft::LayersNumber>& getCells() const;
  const std::vector<Cell>& getCellsInLayer(Int_t layerId) const;
  std::vector<TrackLTF>& getTracksLTF();
  TrackLTF& getCurrentTrackLTF();
  void removeCurrentTrackLTF();
  Road& getCurrentRoad();
  void removeCurrentRoad();
  std::vector<Road>& getRoads();
  std::vector<TrackCA>& getTracksCA();
  TrackCA& getCurrentTrackCA();
  void removeCurrentTrackCA();

  template <typename... T>
  void addClusterToLayer(Int_t layer, T&&... args);

  template <typename... T>
  void addCellToLayer(Int_t layer, T&&... args);

  void addClusterLabelToLayer(Int_t layer, const MCCompLabel label);
  void addClusterExternalIndexToLayer(Int_t layer, const Int_t idx);
  void addClusterBinIndexRangeToLayer(Int_t layer, const std::pair<Int_t, std::pair<Int_t, Int_t>> range);
  void addTrackLTF();
  void addTrackCA();
  void addRoad();

  void initialise();
  void sortClusters();

  Bool_t isClusterUsed(Int_t layer, Int_t clusterId) const;
  void markUsedCluster(Int_t layer, Int_t clusterId);

  void clear();

 private:
  Int_t mROframeId;
  std::array<std::vector<Cluster>, Constants::mft::LayersNumber> mClusters;
  std::array<std::vector<MCCompLabel>, Constants::mft::LayersNumber> mClusterLabels;
  std::array<std::vector<Int_t>, Constants::mft::LayersNumber> mClusterExternalIndices;
  std::array<std::map<Int_t, std::pair<Int_t, Int_t>>, Constants::mft::LayersNumber> mClusterBinIndexRange;
  std::array<std::vector<Bool_t>, Constants::mft::LayersNumber> mUsedClusters;
  std::array<std::vector<Cell>, Constants::mft::LayersNumber> mCells;
  std::vector<TrackLTF> mTracksLTF;
  std::vector<TrackCA> mTracksCA;
  std::vector<Road> mRoads;
};

inline Int_t ROframe::getROFrameId() const { return mROframeId; }

inline const std::array<std::vector<Cluster>, Constants::mft::LayersNumber>& ROframe::getClusters() const
{
  return mClusters;
}

inline const std::vector<Cluster>& ROframe::getClustersInLayer(Int_t layerId) const
{
  return mClusters[layerId];
}

inline const MCCompLabel& ROframe::getClusterLabels(Int_t layerId, const Cluster& cl) const
{
  return mClusterLabels[layerId][cl.clusterId];
}

inline const MCCompLabel& ROframe::getClusterLabels(Int_t layerId, const Int_t clId) const
{
  return mClusterLabels[layerId][clId];
}

inline Int_t ROframe::getClusterExternalIndex(Int_t layerId, const Int_t clId) const
{
  return mClusterExternalIndices[layerId][clId];
}

inline const std::map<Int_t, std::pair<Int_t, Int_t>>& ROframe::getClusterBinIndexRange(Int_t layerId) const
{
  return mClusterBinIndexRange[layerId];
}

inline const std::vector<Cell>& ROframe::getCellsInLayer(Int_t layerId) const
{
  return mCells[layerId];
}

template <typename... T>
void ROframe::addClusterToLayer(Int_t layer, T&&... values)
{
  mClusters[layer].emplace_back(std::forward<T>(values)...);
}

inline void ROframe::addClusterLabelToLayer(Int_t layer, const MCCompLabel label) { mClusterLabels[layer].emplace_back(label); }

inline void ROframe::addClusterExternalIndexToLayer(Int_t layer, const Int_t idx)
{
  mClusterExternalIndices[layer].push_back(idx);
}

inline void ROframe::addClusterBinIndexRangeToLayer(Int_t layer, const std::pair<Int_t, std::pair<Int_t, Int_t>> range)
{
  mClusterBinIndexRange[layer].insert(range);
}

template <typename... T>
void ROframe::addCellToLayer(Int_t layer, T&&... values)
{
  mCells[layer].emplace_back(layer, std::forward<T>(values)...);
}

inline Bool_t ROframe::isClusterUsed(Int_t layer, Int_t clusterId) const
{
  return mUsedClusters[layer][clusterId];
}

inline void ROframe::markUsedCluster(Int_t layer, Int_t clusterId) { mUsedClusters[layer][clusterId] = kTRUE; }

inline void ROframe::addTrackLTF() { mTracksLTF.emplace_back(); }

inline TrackLTF& ROframe::getCurrentTrackLTF()
{
  return mTracksLTF.back();
}

inline void ROframe::removeCurrentTrackLTF()
{
  mTracksLTF.pop_back();
}

inline std::vector<TrackLTF>& ROframe::getTracksLTF()
{
  return mTracksLTF;
}

inline void ROframe::addRoad()
{
  mRoads.emplace_back();
  mRoads.back().setRoadId(mRoads.size() - 1);
}

inline Road& ROframe::getCurrentRoad()
{
  return mRoads.back();
}

inline void ROframe::removeCurrentRoad()
{
  mRoads.pop_back();
}

inline std::vector<Road>& ROframe::getRoads()
{
  return mRoads;
}

inline void ROframe::addTrackCA() { mTracksCA.emplace_back(); }

inline TrackCA& ROframe::getCurrentTrackCA()
{
  return mTracksCA.back();
}

inline void ROframe::removeCurrentTrackCA()
{
  mTracksCA.pop_back();
}

inline std::vector<TrackCA>& ROframe::getTracksCA()
{
  return mTracksCA;
}

inline void ROframe::clear()
{
  for (Int_t iLayer = 0; iLayer < Constants::mft::LayersNumber; ++iLayer) {
    mClusters[iLayer].clear();
    mClusterLabels[iLayer].clear();
    mClusterExternalIndices[iLayer].clear();
    mClusterBinIndexRange[iLayer].clear();
    mCells[iLayer].clear();
  }
  mTracksLTF.clear();
  mTracksCA.clear();
  mRoads.clear();
}

} // namespace MFT
} // namespace o2

#endif /* O2_MFT_ROFRAME_H_ */

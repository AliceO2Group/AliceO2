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

class ROframe
{
 public:
  ROframe(Int_t ROframeId);
  Int_t getROFrameId() const { return mROframeId; }
  Int_t getTotalClusters() const;

  void setROFrameId(const Int_t rofid) { mROframeId = rofid; }

  std::vector<Cluster>& getClustersInLayer(Int_t layerId) { return mClusters[layerId]; }

  const MCCompLabel& getClusterLabels(Int_t layerId, const Int_t clusterId) const { return mClusterLabels[layerId][clusterId]; }

  const std::array<std::pair<Int_t, Int_t>, constants::index_table::MaxRPhiBins>& getClusterBinIndexRange(Int_t layerId) const { return mClusterBinIndexRange[layerId]; }

  const Int_t getClusterExternalIndex(Int_t layerId, const Int_t clusterId) const { return mClusterExternalIndices[layerId][clusterId]; }

  std::vector<TrackLTF>& getTracksLTF();
  TrackLTF& getCurrentTrackLTF();

  std::vector<TrackCA>& getTracksCA();
  TrackCA& getCurrentTrackCA();

  Road& getCurrentRoad();

  template <typename... T>
  void addClusterToLayer(Int_t layer, T&&... args);

  void addClusterLabelToLayer(Int_t layer, const MCCompLabel label);
  void addClusterExternalIndexToLayer(Int_t layer, const Int_t idx);

  void addTrackLTF() { mTracksLTF.emplace_back(); }

  void addTrackCA(const Int_t);

  void addRoad();

  void initialize();

  void sortClusters();

  void clear();

  const Int_t getNClustersInLayer(Int_t layerId) const { return mClusters[layerId].size(); }

 private:
  Int_t mROframeId;
  std::array<std::vector<Cluster>, constants::mft::LayersNumber> mClusters;
  std::array<std::vector<MCCompLabel>, constants::mft::LayersNumber> mClusterLabels;
  std::array<std::vector<Int_t>, constants::mft::LayersNumber> mClusterExternalIndices;
  std::array<std::array<std::pair<Int_t, Int_t>, constants::index_table::MaxRPhiBins>, constants::mft::LayersNumber> mClusterBinIndexRange;
  std::vector<TrackLTF> mTracksLTF;
  std::vector<TrackCA> mTracksCA;
  std::vector<Road> mRoads;
};

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

inline TrackLTF& ROframe::getCurrentTrackLTF()
{
  return mTracksLTF.back();
}

inline std::vector<TrackLTF>& ROframe::getTracksLTF()
{
  return mTracksLTF;
}

inline void ROframe::addRoad()
{
  mRoads.emplace_back();
}

inline Road& ROframe::getCurrentRoad()
{
  return mRoads.back();
}

inline void ROframe::addTrackCA(const Int_t roadId)
{
  mTracksCA.emplace_back();
}

inline TrackCA& ROframe::getCurrentTrackCA()
{
  return mTracksCA.back();
}

inline std::vector<TrackCA>& ROframe::getTracksCA()
{
  return mTracksCA;
}

inline void ROframe::clear()
{
  for (Int_t iLayer = 0; iLayer < constants::mft::LayersNumber; ++iLayer) {
    mClusters[iLayer].clear();
    mClusterLabels[iLayer].clear();
    mClusterExternalIndices[iLayer].clear();
    for (Int_t iBin = 0; iBin < constants::index_table::MaxRPhiBins; ++iBin) {
      mClusterBinIndexRange[iLayer][iBin] = std::pair<Int_t, Int_t>(0, -1);
    }
  }
  mTracksLTF.clear();
  mTracksCA.clear();
  mRoads.clear();
}

} // namespace mft
} // namespace o2

#endif /* O2_MFT_ROFRAME_H_ */

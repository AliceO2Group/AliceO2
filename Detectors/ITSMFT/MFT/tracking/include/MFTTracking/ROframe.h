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

template <typename T>
class ROframe
{
 public:
  void Reserve(int nClusters = 0, float fraction = 0.12f)
  {
    auto layer = constants::mft::LayersNumber;
    while (layer--) {
      mClusters[layer].reserve(nClusters * fraction);
      mClusterExternalIndices[layer].reserve(nClusters * fraction);
      mClusterSizes[layer].reserve(nClusters * fraction);
    }
    mTracks.reserve(nClusters * fraction);
  }
  Int_t getTotalClusters() const;

  std::vector<Cluster>& getClustersInLayer(Int_t layerId) { return mClusters[layerId]; }

  const MCCompLabel& getClusterLabels(Int_t layerId, const Int_t clusterId) const { return mClusterLabels[layerId][clusterId]; }

  const Int_t getClusterExternalIndex(Int_t layerId, const Int_t clusterId) const { return mClusterExternalIndices[layerId][clusterId]; }

  const Int_t getClusterSize(Int_t layerId, const Int_t clusterId) const { return mClusterSizes[layerId][clusterId]; }

  std::vector<T>& getTracks() { return mTracks; }
  T& getCurrentTrack() { return mTracks.back(); }

  Road& getCurrentRoad() { return mRoads.back(); }

  template <typename... C>
  void addClusterToLayer(Int_t layer, C&&... args)
  {
    mClusters[layer].emplace_back(std::forward<C>(args)...);
  }
  void addClusterLabelToLayer(Int_t layer, const MCCompLabel label) { mClusterLabels[layer].emplace_back(label); }
  void addClusterExternalIndexToLayer(Int_t layer, const Int_t idx) { mClusterExternalIndices[layer].push_back(idx); }
  void addClusterSizeToLayer(Int_t layer, const Int_t clusterSize) { mClusterSizes[layer].push_back(clusterSize); }

  void addTrack(bool isCA = false)
  {
    mTracks.emplace_back(isCA);
  }

  void addRoad() { mRoads.emplace_back(); }

  void sortClusters();

  void clear()
  {

    for (Int_t iLayer = 0; iLayer < constants::mft::LayersNumber; ++iLayer) {
      mClusters[iLayer].clear();
      mClusterLabels[iLayer].clear();
      mClusterExternalIndices[iLayer].clear();
      mClusterSizes[iLayer].clear();
    }
    mTracks.clear();
    mRoads.clear();
  }

  const Int_t getNClustersInLayer(Int_t layerId) const { return mClusters[layerId].size(); }

 private:
  std::array<std::vector<Cluster>, constants::mft::LayersNumber> mClusters;
  std::array<std::vector<MCCompLabel>, constants::mft::LayersNumber> mClusterLabels;
  std::array<std::vector<Int_t>, constants::mft::LayersNumber> mClusterExternalIndices;
  std::array<std::vector<Int_t>, constants::mft::LayersNumber> mClusterSizes;
  std::vector<T> mTracks;
  std::vector<Road> mRoads;
};

} // namespace mft
} // namespace o2

#endif /* O2_MFT_ROFRAME_H_ */

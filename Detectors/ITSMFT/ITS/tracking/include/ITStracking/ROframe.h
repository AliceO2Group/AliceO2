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
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_ROFRAME_H_
#define TRACKINGITSU_INCLUDE_ROFRAME_H_

#include <array>
#include <vector>
#include <utility>
#include <cassert>
#include <gsl/gsl>

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"

#include "ReconstructionDataFormats/Vertex.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace its
{

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

class ROframe final
{
 public:
  ROframe(int ROframeId);
  int getROFrameId() const;
  const float3& getPrimaryVertex(const int) const;
  int getPrimaryVerticesNum() const;
  void addPrimaryVertex(const float, const float, const float);
  void addPrimaryVertices(std::vector<Vertex> vertices);
  void addPrimaryReconstructedVertex(const float, const float, const float);
  void printPrimaryVertices() const;
  int getTotalClusters() const;
  bool empty() const;

  const std::array<std::vector<Cluster>, constants::its::LayersNumber>& getClusters() const;
  const std::vector<Cluster>& getClustersOnLayer(int layerId) const;
  const std::vector<TrackingFrameInfo>& getTrackingFrameInfoOnLayer(int layerId) const;
  const std::array<std::vector<TrackingFrameInfo>, constants::its::LayersNumber>& getTrackingFrameInfo() const;

  const TrackingFrameInfo& getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const;
  const MCCompLabel& getClusterLabels(int layerId, const Cluster& cl) const;
  const MCCompLabel& getClusterLabels(int layerId, const int clId) const;
  int getClusterExternalIndex(int layerId, const int clId) const;
  std::vector<int> getTracksId(const int layerId, const std::vector<Cluster>& cl);

  template <typename... T>
  void addClusterToLayer(int layer, T&&... args);
  template <typename... T>
  void addTrackingFrameInfoToLayer(int layer, T&&... args);
  void addClusterLabelToLayer(int layer, const MCCompLabel label);
  void addClusterExternalIndexToLayer(int layer, const int idx);
  bool hasMCinformation() const;

  void clear();

 private:
  const int mROframeId;
  std::vector<float3> mPrimaryVertices;
  std::array<std::vector<Cluster>, constants::its::LayersNumber> mClusters;
  std::array<std::vector<TrackingFrameInfo>, constants::its::LayersNumber> mTrackingFrameInfo;
  std::array<std::vector<MCCompLabel>, constants::its::LayersNumber> mClusterLabels;
  std::array<std::vector<int>, constants::its::LayersNumber> mClusterExternalIndices;
};

inline int ROframe::getROFrameId() const { return mROframeId; }

inline const float3& ROframe::getPrimaryVertex(const int vertexIndex) const { return mPrimaryVertices[vertexIndex]; }

inline int ROframe::getPrimaryVerticesNum() const { return mPrimaryVertices.size(); }

inline bool ROframe::empty() const { return getTotalClusters() == 0; }

inline const std::array<std::vector<Cluster>, constants::its::LayersNumber>& ROframe::getClusters() const
{
  return mClusters;
}

inline const std::vector<Cluster>& ROframe::getClustersOnLayer(int layerId) const
{
  return mClusters[layerId];
}

inline const std::vector<TrackingFrameInfo>& ROframe::getTrackingFrameInfoOnLayer(int layerId) const
{
  return mTrackingFrameInfo[layerId];
}

inline const std::array<std::vector<TrackingFrameInfo>, constants::its::LayersNumber>& ROframe::getTrackingFrameInfo() const
{
  return mTrackingFrameInfo;
}

inline const TrackingFrameInfo& ROframe::getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const
{
  return mTrackingFrameInfo[layerId][cl.clusterId];
}

inline const MCCompLabel& ROframe::getClusterLabels(int layerId, const Cluster& cl) const
{
  return mClusterLabels[layerId][cl.clusterId];
}

inline const MCCompLabel& ROframe::getClusterLabels(int layerId, const int clId) const
{
  return mClusterLabels[layerId][clId];
}

inline int ROframe::getClusterExternalIndex(int layerId, const int clId) const
{
  return mClusterExternalIndices[layerId][clId];
}

inline std::vector<int> ROframe::getTracksId(const int layerId, const std::vector<Cluster>& cl)
{
  std::vector<int> tracksId;
  for (auto& cluster : cl) {
    tracksId.push_back(getClusterLabels(layerId, cluster).isNoise() ? -1 : getClusterLabels(layerId, cluster).getTrackID());
  }
  return tracksId;
}

template <typename... T>
void ROframe::addClusterToLayer(int layer, T&&... values)
{
  mClusters[layer].emplace_back(std::forward<T>(values)...);
}

template <typename... T>
void ROframe::addTrackingFrameInfoToLayer(int layer, T&&... values)
{
  mTrackingFrameInfo[layer].emplace_back(std::forward<T>(values)...);
}

inline void ROframe::addClusterLabelToLayer(int layer, const MCCompLabel label) { mClusterLabels[layer].emplace_back(label); }

inline void ROframe::addClusterExternalIndexToLayer(int layer, const int idx)
{
  mClusterExternalIndices[layer].push_back(idx);
}

inline void ROframe::clear()
{
  for (int iL = 0; iL < constants::its::LayersNumber; ++iL) {
    mClusters[iL].clear();
    mTrackingFrameInfo[iL].clear();
    mClusterLabels[iL].clear();
    mClusterExternalIndices[iL].clear();
  }
  mPrimaryVertices.clear();
}

inline bool ROframe::hasMCinformation() const
{
  for (const auto& vect : mClusterLabels) {
    if (!vect.empty()) {
      return true;
    }
  }
  return false;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_ROFRAME_H_ */

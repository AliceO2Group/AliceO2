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
  ROframe(int ROframeId, int nLayers);
  int getROFrameId() const;
  const float3& getPrimaryVertex(const int) const;
  int getPrimaryVerticesNum() const;
  void addPrimaryVertex(const float, const float, const float);
  void addPrimaryVertices(std::vector<Vertex> vertices);
  void addPrimaryReconstructedVertex(const float, const float, const float);
  void printPrimaryVertices() const;
  int getTotalClusters() const;
  bool empty() const;

  const auto& getClusters() const { return mClusters; }
  const std::vector<Cluster>& getClustersOnLayer(int layerId) const;
  const std::vector<TrackingFrameInfo>& getTrackingFrameInfoOnLayer(int layerId) const;
  const auto& getTrackingFrameInfo() const { return mTrackingFrameInfo; }

  const TrackingFrameInfo& getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const;
  const MCCompLabel& getClusterFirstLabel(int layerId, const Cluster& cl) const;
  const MCCompLabel& getClusterFirstLabel(int layerId, const int clId) const;
  const gsl::span<const o2::MCCompLabel> getClusterLabels(int layerId, const int clId) const;
  const gsl::span<const o2::MCCompLabel> getClusterLabels(int layerId, const Cluster& cl) const;
  int getClusterExternalIndex(int layerId, const int clId) const;
  std::vector<int> getTracksId(const int layerId, const std::vector<Cluster>& cl);

  template <typename... T>
  void addClusterToLayer(int layer, T&&... args);
  template <typename... T>
  void addTrackingFrameInfoToLayer(int layer, T&&... args);
  void setMClabelsContainer(const dataformats::MCTruthContainer<MCCompLabel>* ptr);
  void addClusterExternalIndexToLayer(int layer, const int idx);
  bool hasMCinformation() const;

  void clear();

 private:
  const int mROframeId;
  const o2::dataformats::MCTruthContainer<MCCompLabel>* mMClabels = nullptr;
  std::vector<float3> mPrimaryVertices;
  std::vector<std::vector<Cluster>> mClusters;
  std::vector<std::vector<TrackingFrameInfo>> mTrackingFrameInfo;
  std::vector<std::vector<int>> mClusterExternalIndices;
};

inline int ROframe::getROFrameId() const { return mROframeId; }

inline const float3& ROframe::getPrimaryVertex(const int vertexIndex) const { return mPrimaryVertices[vertexIndex]; }

inline int ROframe::getPrimaryVerticesNum() const { return mPrimaryVertices.size(); }

inline bool ROframe::empty() const { return getTotalClusters() == 0; }

inline const std::vector<Cluster>& ROframe::getClustersOnLayer(int layerId) const
{
  return mClusters[layerId];
}

inline const std::vector<TrackingFrameInfo>& ROframe::getTrackingFrameInfoOnLayer(int layerId) const
{
  return mTrackingFrameInfo[layerId];
}

inline const TrackingFrameInfo& ROframe::getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const
{
  return mTrackingFrameInfo[layerId][cl.clusterId];
}

inline const MCCompLabel& ROframe::getClusterFirstLabel(int layerId, const Cluster& cl) const
{
  return getClusterFirstLabel(layerId, cl.clusterId);
}

inline const MCCompLabel& ROframe::getClusterFirstLabel(int layerId, const int clId) const
{
  return *(mMClabels->getLabels(getClusterExternalIndex(layerId, clId)).begin());
}

inline const gsl::span<const o2::MCCompLabel> ROframe::getClusterLabels(int layerId, const int clId) const
{
  return mMClabels->getLabels(getClusterExternalIndex(layerId, clId));
}

inline const gsl::span<const o2::MCCompLabel> ROframe::getClusterLabels(int layerId, const Cluster& cl) const
{
  return getClusterLabels(layerId, cl.clusterId);
}

inline int ROframe::getClusterExternalIndex(int layerId, const int clId) const
{
  return mClusterExternalIndices[layerId][clId];
}

inline std::vector<int> ROframe::getTracksId(const int layerId, const std::vector<Cluster>& cl)
{
  std::vector<int> tracksId;
  for (auto& cluster : cl) {
    tracksId.push_back(getClusterFirstLabel(layerId, cluster).isNoise() ? -1 : getClusterFirstLabel(layerId, cluster).getTrackID());
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

inline void ROframe::setMClabelsContainer(const dataformats::MCTruthContainer<MCCompLabel>* ptr)
{
  mMClabels = ptr;
}

inline void ROframe::addClusterExternalIndexToLayer(int layer, const int idx)
{
  mClusterExternalIndices[layer].push_back(idx);
}

inline void ROframe::clear()
{
  for (unsigned int iL = 0; iL < mClusters.size(); ++iL) {
    mClusters[iL].clear();
    mTrackingFrameInfo[iL].clear();
    // mClusterLabels[iL].clear();
    mClusterExternalIndices[iL].clear();
  }
  mPrimaryVertices.clear();
  mMClabels = nullptr;
}

inline bool ROframe::hasMCinformation() const
{
  // for (const auto& vect : mClusterLabels) {
  //   if (!vect.empty()) {
  //     return true;
  //   }
  // }
  // return false;
  return mMClabels;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_ROFRAME_H_ */

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

#ifndef TRACKINGITSU_INCLUDE_ROframe_H_
#define TRACKINGITSU_INCLUDE_ROframe_H_

#include <array>
#include <vector>
#include <utility>
#include <cassert>
#include <gsl/gsl>

#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace ITS
{

class ROframe final
{
 public:
  ROframe(int ROframeId);
  int getROFrameId() const;
  const float3& getPrimaryVertex(const int) const;
  int getPrimaryVerticesNum() const;
  void addPrimaryVertex(const float, const float, const float);
  void addPrimaryReconstructedVertex(const float, const float, const float);
  void printPrimaryVertices() const;
  int getTotalClusters() const;

  const std::vector<Cluster>& getClustersOnLayer(int layerId) const;
  const std::vector<TrackingFrameInfo>& getTrackingFrameInfoOnLayer(int layerId) const;

  const TrackingFrameInfo& getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const;
  const MCCompLabel& getClusterLabels(int layerId, const Cluster& cl) const ;
  const MCCompLabel& getClusterLabels(int layerId, const int clId) const;

  template <typename... T>
  void addClusterToLayer(int layer, T&&... args);
  template <typename... T>
  void addTrackingFrameInfoToLayer(int layer, T&&... args);
  void addClusterLabelToLayer(int layer, const MCCompLabel label);

  void clear();

 private:
  const int mROframeId;
  std::vector<float3> mPrimaryVertices;
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumber> mClusters;
  std::array<std::vector<TrackingFrameInfo>, Constants::ITS::LayersNumber> mTrackingFrameInfo;
  std::array<std::vector<MCCompLabel>, Constants::ITS::LayersNumber> mClusterLabels;
};

inline int ROframe::getROFrameId() const { return mROframeId; }

inline const float3& ROframe::getPrimaryVertex(const int vertexIndex) const { return mPrimaryVertices[vertexIndex]; }

inline int ROframe::getPrimaryVerticesNum() const { return mPrimaryVertices.size(); }

inline const std::vector<Cluster>& ROframe::getClustersOnLayer(int layerId) const {
  return mClusters[layerId];
}

inline const std::vector<TrackingFrameInfo>& ROframe::getTrackingFrameInfoOnLayer(int layerId) const {
  return mTrackingFrameInfo[layerId];
}

inline const TrackingFrameInfo& ROframe::getClusterTrackingFrameInfo(int layerId, const Cluster& cl) const {
  return mTrackingFrameInfo[layerId][cl.clusterId];
}

inline const MCCompLabel& ROframe::getClusterLabels(int layerId, const Cluster& cl) const{
  return mClusterLabels[layerId][cl.clusterId];
}

inline const MCCompLabel& ROframe::getClusterLabels(int layerId, const int clId) const {
  return mClusterLabels[layerId][clId];
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

inline void ROframe::clear()
{
  for (int iL = 0; iL < Constants::ITS::LayersNumber; ++iL) {
    mClusters[iL].clear();
    mTrackingFrameInfo[iL].clear();
    mClusterLabels[iL].clear();
  }
  mPrimaryVertices.clear();
}

} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_ROframe_H_ */

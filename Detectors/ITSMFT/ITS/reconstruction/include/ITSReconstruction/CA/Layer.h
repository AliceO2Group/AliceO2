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
/// \file Layer.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_LAYER_H_
#define TRACKINGITSU_INCLUDE_LAYER_H_

#include <vector>
#include <utility>

#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Definitions.h"

namespace o2
{
namespace ITS
{
namespace CA
{

class Layer final
{
 public:
  Layer();
  Layer(const int layerIndex);

  int getLayerIndex() const;
  const std::vector<Cluster>& getClusters() const;
  const std::vector<TrackingFrameInfo>& getTrackingFrameInfo() const;
  const Cluster& getCluster(int idx) const;
  const TrackingFrameInfo& getTrackingFrameInfo(int idx) const;
  int getClustersSize() const;
  template <typename... T>
  void addCluster(T&&... args);
  template <typename... T>
  void addTrackingFrameInfo(T&&... args);

  void clear();

 private:
  int mLayerIndex;
  std::vector<Cluster> mClusters;
  std::vector<TrackingFrameInfo> mTrackingFrameInfo;
};

inline int Layer::getLayerIndex() const { return mLayerIndex; }

inline const std::vector<Cluster>& Layer::getClusters() const { return mClusters; }

inline const std::vector<TrackingFrameInfo>& Layer::getTrackingFrameInfo() const { return mTrackingFrameInfo; }

inline const Cluster& Layer::getCluster(int clusterIndex) const { return mClusters[clusterIndex]; }

inline const TrackingFrameInfo& Layer::getTrackingFrameInfo(int clusterIndex) const
{
  return mTrackingFrameInfo[clusterIndex];
}

inline int Layer::getClustersSize() const { return mClusters.size(); }

template <typename... T>
void Layer::addCluster(T&&... args)
{
  mClusters.emplace_back(std::forward<T>(args)...);
}

template <typename... T>
void Layer::addTrackingFrameInfo(T&&... args)
{
  mTrackingFrameInfo.emplace_back(std::forward<T>(args)...);
}

inline void Layer::clear()
{
  mClusters.clear();
  mTrackingFrameInfo.clear();
}
}
}
}

#endif /* TRACKINGITSU_INCLUDE_LAYER_H_ */

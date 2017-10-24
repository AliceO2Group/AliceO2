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

class Layer
  final
  {
    public:
      Layer();
      Layer(const int);

      int getLayerIndex() const;
      const std::vector<Cluster>& getClusters() const;
      const Cluster& getCluster(int) const;
      int getClustersSize() const;

      const std::vector<TrackingFrameInfo>& getTrackingFrameInfo() const;
      const TrackingFrameInfo& getTrackingFrameInfo(int) const;

      void addCluster(const int, const float, const float, const float, const float, const int);
      template<typename ... T> void addCluster(T&& ... values);
      template<typename ... T> void addTrackingFrameInfo(T&& ... values);

      void clear();

    private:
      int mLayerIndex;
      std::vector<Cluster> mClusters;
      std::vector<TrackingFrameInfo> mTrackingFrameInfo;
  };

  inline int Layer::getLayerIndex() const
  {
    return mLayerIndex;
  }

  inline const std::vector<Cluster>& Layer::getClusters() const
  {
    return mClusters;
  }

  inline const Cluster& Layer::getCluster(int clusterIndex) const
  {
    return mClusters[clusterIndex];
  }

  inline int Layer::getClustersSize() const
  {
    return mClusters.size();
  }

  inline const std::vector<TrackingFrameInfo>& Layer::getTrackingFrameInfo() const
  { 
    return mTrackingFrameInfo;
  }
  
  inline const TrackingFrameInfo& Layer::getTrackingFrameInfo(int index) const {
    return mTrackingFrameInfo[index];
  }

  template<typename ... T> void Layer::addCluster(T&& ... values) {
    mClusters.emplace_back(std::forward<T>(values)...);
  }

  template<typename ... T> void Layer::addTrackingFrameInfo(T&& ... values) {
    mTrackingFrameInfo.emplace_back(std::forward<T>(values)...);
  }
}
}
}

#endif /* TRACKINGITSU_INCLUDE_LAYER_H_ */

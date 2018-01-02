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
/// \file Event.h
/// \brief 
///

#ifndef TRACKINGITSU_INCLUDE_EVENT_H_
#define TRACKINGITSU_INCLUDE_EVENT_H_

#include <array>
#include <vector>
#include <utility>
#include <cassert>
#include <gsl/gsl>

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Layer.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace ITS
{
namespace CA
{

class Event
  final
  {
    public:
      explicit Event(const int=0);

      int getEventId() const;
      const float3& getPrimaryVertex(const int) const;
      const Layer& getLayer(const int) const;
      int getPrimaryVerticesNum() const;
      void addPrimaryVertex(const float, const float, const float);
      void printPrimaryVertices() const;
      int getTotalClusters() const;

      template<typename... T> void addClusterToLayer(int layer, T&&... args);
      template<typename... T> void addTrackingFrameInfoToLayer(int layer, T&&... args);

      void clear();
      const gsl::span<const MCCompLabel> getClusterMClabels(int layer, int clustId) const;
      int  getGlobalIndex(int layer, int clusId) const;
      void setClusterMCtruth(const dataformats::MCTruthContainer<MCCompLabel> *clsLabels);
    private:
      const int mEventId;
      std::vector<float3> mPrimaryVertices;
      std::array<Layer, Constants::ITS::LayersNumber> mLayers;
      const dataformats::MCTruthContainer<MCCompLabel> *mClsLabels;
  };

  inline int Event::getEventId() const
  {
    return mEventId;
  }

  inline const float3& Event::getPrimaryVertex(const int vertexIndex) const
  {
    return mPrimaryVertices[vertexIndex];
  }

  inline const Layer& Event::getLayer(const int layerIndex) const
  {
    assert (layerIndex >= 0 && layerIndex < Constants::ITS::LayersNumber);
    return mLayers[layerIndex];
  }

  inline int Event::getPrimaryVerticesNum() const
  {
    return mPrimaryVertices.size();
  }

  template<typename ... T> void Event::addClusterToLayer(int layer, T&& ... values)
  {
    mLayers[layer].addCluster(std::forward<T>(values)...);
  }

  template<typename ... T> void Event::addTrackingFrameInfoToLayer(int layer, T&& ... values)
  {
    mLayers[layer].addTrackingFrameInfo(std::forward<T>(values)...);
  }

  inline void Event::clear()
  {
    for (auto& layer : mLayers)
    {
      layer.clear();
    }
    mPrimaryVertices.clear();
  }

  inline const gsl::span<const MCCompLabel> Event::getClusterMClabels(int layer, int clustId) const
  {
    return mClsLabels->getLabels(getGlobalIndex(layer,clustId));
  }

  inline int Event::getGlobalIndex(int layer, int index) const
  {
    for (int iLayer{0}; iLayer < layer; ++iLayer)
    {
      index += mLayers[iLayer].getClustersSize();
    }
    return index;
  }

  inline void Event::setClusterMCtruth(const dataformats::MCTruthContainer<MCCompLabel> *clsLabels)
  {
    mClsLabels = clsLabels;
  }


}
}
}

#endif /* TRACKINGITSU_INCLUDE_EVENT_H_ */

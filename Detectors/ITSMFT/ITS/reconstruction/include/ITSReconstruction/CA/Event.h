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

#include <gsl/span>

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
      explicit Event(const int, const float mBz = 0.5f);

      int getEventId() const;
      const float3& getPrimaryVertex(const int) const;
      const Layer& getLayer(const int) const;
      int getPrimaryVerticesNum() const;
      void addPrimaryVertex(const float, const float, const float);
      void printPrimaryVertices() const;
      int getTotalClusters() const;

      const gsl::span<const MCCompLabel> getClusterMClabel(int layer, int cluster) const;
      void setMCTruthContainers(const dataformats::MCTruthContainer<MCCompLabel> *cls);

      template<typename ... T> void addCluster(int layer, T&& ... values);
      template<typename ... T> void addTrackingFrameInfo(int layer, T&& ... values);
      float getBz() const;
      void  clear();

    private:
      const int mEventId;
      const float mBz;
      std::vector<float3> mPrimaryVertices;
      std::array<Layer, Constants::ITS::LayersNumber> mLayers;
      std::array<int, Constants::ITS::LayersNumber - 1> mIdOffSets;
      const dataformats::MCTruthContainer<MCCompLabel> *mClsLabels = nullptr;
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
    return mLayers[layerIndex];
  }

  inline int Event::getPrimaryVerticesNum() const
  {

    return mPrimaryVertices.size();
  }

  inline float Event::getBz() const
  {
    return mBz;
  }

  template<typename ... T> void Event::addCluster(int layer, T&& ... values)
  {
    mLayers[layer].addCluster(std::forward<T>(values)...);
  }

  template<typename ... T> void Event::addTrackingFrameInfo(int layer, T&& ... values) {
    mLayers[layer].addTrackingFrameInfo(std::forward<T>(values)...);
  }

  inline void Event::setMCTruthContainers(const dataformats::MCTruthContainer<MCCompLabel> *cls) {
    mClsLabels = cls;
  }

  inline const gsl::span<const MCCompLabel> Event::getClusterMClabel(int layer, int cluster) const {
    return mClsLabels->getLabels((layer ? mIdOffSets[layer] : 0) + cluster);
  }

}
}
}

#endif /* TRACKINGITSU_INCLUDE_EVENT_H_ */

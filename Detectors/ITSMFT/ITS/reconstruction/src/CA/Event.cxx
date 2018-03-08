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
/// \file Event.cxx
/// \brief
///

#include "ITSReconstruction/CA/Event.h"

#include <iostream>

namespace o2
{
namespace ITS
{
namespace CA
{

Event::Event(const int eventId) : mEventId{ eventId }
{
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    mLayers[iLayer] = Layer(iLayer);
  }
}

void Event::addPrimaryVertex(const float xCoordinate, const float yCoordinate, const float zCoordinate)
{
  mPrimaryVertices.emplace_back(float3{ xCoordinate, yCoordinate, zCoordinate });
}

void Event::printPrimaryVertices() const
{
  const int verticesNum{ static_cast<int>(mPrimaryVertices.size()) };

  for (int iVertex{ 0 }; iVertex < verticesNum; ++iVertex) {

    const float3& currentVertex = mPrimaryVertices[iVertex];

    std::cout << "-1\t" << currentVertex.x << "\t" << currentVertex.y << "\t" << currentVertex.z << std::endl;
  }
}

int Event::getTotalClusters() const
{
  int totalClusters{ 0 };

  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    totalClusters += mLayers[iLayer].getClustersSize();
  }

  return totalClusters;
}
}
}
}

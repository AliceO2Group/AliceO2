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
/// \file    VisualisationCluster.cxx
/// \author  Julian Myrcha
///

#include "EventVisualisationDataConverter/VisualisationCluster.h"
#include <iostream>

using namespace std;

namespace o2
{
namespace event_visualisation
{

VisualisationCluster::VisualisationCluster(double XYZ[])
{
  setCoordinates(XYZ);
}

void VisualisationCluster::setCoordinates(double xyz[3])
{
  for (int i = 0; i < 3; i++) {
    mCoordinates[i] = xyz[i];
  }
}

VisualisationCluster::VisualisationCluster(rapidjson::Value& tree)
{
  rapidjson::Value& jsonX = tree["X"];
  rapidjson::Value& jsonY = tree["Y"];
  rapidjson::Value& jsonZ = tree["Z"];

  this->mCoordinates[0] = jsonX.GetDouble();
  this->mCoordinates[1] = jsonY.GetDouble();
  this->mCoordinates[2] = jsonZ.GetDouble();
}

rapidjson::Value VisualisationCluster::jsonTree(rapidjson::MemoryPoolAllocator<>& allocator)
{
  rapidjson::Value tree(rapidjson::kObjectType);
  rapidjson::Value jsonX(rapidjson::kNumberType);
  rapidjson::Value jsonY(rapidjson::kNumberType);
  rapidjson::Value jsonZ(rapidjson::kNumberType);
  jsonX.SetDouble(mCoordinates[0]);
  jsonY.SetDouble(mCoordinates[1]);
  jsonZ.SetDouble(mCoordinates[2]);
  tree.AddMember("X", jsonX, allocator);
  tree.AddMember("Y", jsonY, allocator);
  tree.AddMember("Z", jsonZ, allocator);
  return tree;
}

} // namespace event_visualisation
} // namespace o2

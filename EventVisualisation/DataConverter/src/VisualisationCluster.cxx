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

VisualisationCluster::VisualisationCluster(float XYZ[], float time)
{
  setCoordinates(XYZ);
  this->mTime = time;
}

void VisualisationCluster::setCoordinates(float xyz[3])
{
  for (int i = 0; i < 3; i++) {
    mCoordinates[i] = xyz[i];
  }
}

} // namespace event_visualisation
} // namespace o2

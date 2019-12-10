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
  for (int i = 0; i < 3; i++)
    mCoordinates[i] = xyz[i];
}

} // namespace event_visualisation
} // namespace o2

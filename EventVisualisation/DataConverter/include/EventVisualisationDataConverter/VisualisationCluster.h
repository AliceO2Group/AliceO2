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
/// \file    VisualisationCluster.h
/// \author  Julian Myrcha
///

#ifndef ALICE_O2_DATACONVERTER_VISUALISATIONCLUSTER_H
#define ALICE_O2_DATACONVERTER_VISUALISATIONCLUSTER_H

#include "EventVisualisationDataConverter/VisualisationTrack.h"
#include "rapidjson/document.h"

#include <vector>
#include <ctime>

namespace o2
{
namespace event_visualisation
{

/// Minimalistic description of a cluster
///
/// This class is used mainly for visualisation purposes.
/// It stores simple information about clusters, which can be used for visualisation
/// or exported for external applications.

class VisualisationCluster
{
 public:
  VisualisationCluster(rapidjson::Value& tree);
  rapidjson::Value jsonTree(rapidjson::Document::AllocatorType& allocator);

  // Default constructor
  VisualisationCluster(double XYZ[]);

  double X() const { return mCoordinates[0]; }
  double Y() const { return mCoordinates[1]; }
  double Z() const { return mCoordinates[2]; }

 private:
  void setCoordinates(double xyz[3]);
  double mCoordinates[3]; /// Vector of cluster's coordinates

 private:
};
} // namespace event_visualisation
} // namespace o2
#endif // ALICE_O2_DATACONVERTER_VISUALISATIONCLUSTER_H
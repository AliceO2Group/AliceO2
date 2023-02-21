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

/// \file TopologyDictionary.h
/// \brief Definition of the ClusterTopology class for ITS3

#ifndef ALICEO2_ITS3_TOPOLOGYDICTIONARY_H
#define ALICEO2_ITS3_TOPOLOGYDICTIONARY_H

#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITS3/CompCluster.h"

namespace o2
{
namespace its3
{

class TopologyDictionary : public itsmft::TopologyDictionary
{
 public:
  TopologyDictionary(itsmft::TopologyDictionary top) : itsmft::TopologyDictionary{top} {}

  /// Returns the local position of a compact cluster
  math_utils::Point3D<float> getClusterCoordinates(const its3::CompClusterExt& cl) const;
  /// Returns the local position of a compact cluster
  static math_utils::Point3D<float> getClusterCoordinates(const its3::CompClusterExt& cl, const itsmft::ClusterPattern& patt, bool isGroup = true);
};
} // namespace its3
} // namespace o2

#endif

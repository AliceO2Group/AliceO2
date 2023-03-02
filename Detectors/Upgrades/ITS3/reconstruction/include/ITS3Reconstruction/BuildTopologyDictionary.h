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
/// \brief Definition of the BuildTopologyDictionary class for ITS3

#ifndef ALICEO2_ITS3_BUILDTOPOLOGYDICTIONARY_H
#define ALICEO2_ITS3_BUILDTOPOLOGYDICTIONARY_H

#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITS3Base/SuperAlpideParams.h"
namespace o2
{
namespace its3
{

class BuildTopologyDictionary : public itsmft::BuildTopologyDictionary
{
 public:
  /// Updates the information of the found cluster topology
  /// \param cluster cluster topology whose information is updatet
  /// \param dX hit - COG positions along x, expressed in fraction of pixel pitches
  /// \param dZ hit - COG positions along z, expressed in fraction of pixel pitches
  void accountTopology(const itsmft::ClusterTopology& cluster, float dX = IgnoreVal, float dZ = IgnoreVal);

  /// Creates entries for common topologies and groups of rare topologies
  void groupRareTopologies();
};
} // namespace its3
} // namespace o2

#endif

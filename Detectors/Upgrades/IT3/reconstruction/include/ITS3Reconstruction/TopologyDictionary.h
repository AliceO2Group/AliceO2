// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DataFormatsITSMFT/CompCluster.h"

namespace o2
{
namespace its3
{

class TopologyDictionary : public itsmft::TopologyDictionary
{
 public:

  ///Returns the local position of a compact cluster
  math_utils::Point3D<float> getClusterCoordinates(int detID, const itsmft::CompCluster& cl) const;
  ///Returns the local position of a compact cluster
  static math_utils::Point3D<float> getClusterCoordinates(int detID, const itsmft::CompCluster& cl, const itsmft::ClusterPattern& patt, bool isGroup = true);  

}; 
} // namespace its3
} // namespace o2

#endif

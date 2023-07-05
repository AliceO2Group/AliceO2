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

/// @file   TrackAndClusterFilterSpec.h
/// @author Jens Wiechula
/// @brief  track and cluster filtering

#ifndef TPC_TrackAndClusterFilterSpec_H_
#define TPC_TrackAndClusterFilterSpec_H_

#include <string>
#include "Framework/DataProcessorSpec.h"

namespace o2::tpc
{

/// create a processor spec
/// read simulated TPC clusters from file and publish
o2::framework::DataProcessorSpec getTrackAndClusterFilterSpec(const std::string dataDescriptionStr = "TRACKS", const bool writeMC = false);

} // namespace o2::tpc

#endif // TPC_TrackAndClusterFilterSpec_H_

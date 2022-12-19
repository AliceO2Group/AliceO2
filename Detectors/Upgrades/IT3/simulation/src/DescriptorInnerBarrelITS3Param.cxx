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

/// \file DescriptorInnerBarrelITS3Param.h
/// \brief Implementation of the DescriptorInnerBarrelITS3Param class

#include "ITS3Simulation/DescriptorInnerBarrelITS3Param.h"
O2ParamImpl(o2::its3::DescriptorInnerBarrelITS3Param);

namespace o2
{
namespace its3
{

namespace
{
static const std::string confstrings[5] = {"", "ThreeLayersNoDeadZones", "ThreeLayers", "FourLayers", "FiveLayers"};
}

std::string const& DescriptorInnerBarrelITS3Param::getITS3LayerConfigString() const
{
  return confstrings[(int)mVersion];
}

} // namespace its3
} // end namespace o2

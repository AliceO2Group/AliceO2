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

#include <boost/property_tree/ptree.hpp>

#include "Framework/Logger.h"
#include "ITStracking/TrackingConfigParam.h"

O2ParamImpl(o2::its::VertexerParamConfig);
O2ParamImpl(o2::its::TrackerParamConfig);
O2ParamImpl(o2::its::ITSGpuTrackingParamConfig);

namespace o2::its
{

void ITSGpuTrackingParamConfig::maybeOverride() const
{
  if (nBlocks == OverrideValue && nThreads == OverrideValue) {
    return;
  }
  const auto name = getName();
  auto members = getDataMembers();
  for (auto member : *members) {
    if (!member.name.ends_with(BlocksName) && !member.name.ends_with(ThreadsName)) {
      if (nBlocks != OverrideValue && member.name.starts_with(BlocksName) && (member.value != nBlocks)) {
        o2::conf::ConfigurableParam::setValue<int>(name, member.name, nBlocks);
      }
      if (nThreads != OverrideValue && member.name.starts_with(ThreadsName) && (member.value != nThreads)) {
        o2::conf::ConfigurableParam::setValue<int>(name, member.name, nThreads);
      }
    }
  }
  LOGP(info, "Overwriting gpu threading parameters");
} // namespace o2::its

} // namespace o2::its

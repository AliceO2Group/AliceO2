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

#include "CCDBHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"

namespace o2::framework
{

AlgorithmSpec CCDBHelpers::fetchFromCCDB()
{
  return adaptStateful([](DeviceSpec const& spec) { 
      for (auto &route : spec.outputs) {
        if (route.matcher.lifetime != Lifetime::Condition) {
          continue;
        }
        LOGP(info, "The following route is a condition {}", route.matcher);
        for (auto& metadata : route.matcher.metadata) {
          LOGP(info, "- {}", metadata.name);
        }
      }
      
      return adaptStateless([](DataAllocator& ctx) {
                              }); });
}

} // namespace o2::framework

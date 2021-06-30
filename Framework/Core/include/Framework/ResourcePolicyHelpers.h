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

#ifndef O2_FRAMEWORK_RESOURCEPOLICYHELPERS_H_
#define O2_FRAMEWORK_RESOURCEPOLICYHELPERS_H_

#include "Framework/ResourcePolicy.h"
#include <functional>
#include <string>

namespace o2::framework
{

struct ResourcePolicyHelpers {
  static ResourcePolicy trivialTask(char const* taskMatcher);
  static ResourcePolicy cpuBoundTask(char const* taskMatcher, int maxCPUs = 1);
  static ResourcePolicy sharedMemoryBoundTask(char const* taskMatcher, int maxMemory);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_RESOURCEPOLICYHELPERS_H_

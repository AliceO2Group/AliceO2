// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ResourcePolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "ResourcesMonitoringHelper.h"

#include <string>
#include <regex>

namespace o2::framework
{

/// A trivial task is a task which will execute regardless of
/// the resources available.
ResourcePolicy ResourcePolicyHelpers::trivialTask(char const* s)
{
  return ResourcePolicy{
    "trivial",
    [matcher = std::regex(s)](DeviceSpec const& spec) -> bool {
      return std::regex_match(spec.name, matcher);
    },
    [](ComputingQuotaOffer const&) { return 127; }};
}

ResourcePolicy ResourcePolicyHelpers::cpuBoundTask(char const* s, int maxCPUs)
{
  return ResourcePolicy{
    "cpu-bound",
    [matcher = std::regex(s)](DeviceSpec const& spec) -> bool {
      return std::regex_match(spec.name, matcher);
    },
    [maxCPUs](ComputingQuotaOffer const& offer) -> int8_t { return offer.cpu >= maxCPUs ? 127 : 0; }};
}

ResourcePolicy ResourcePolicyHelpers::sharedMemoryBoundTask(char const* s, int maxSharedMemory)
{
  return ResourcePolicy{
    "shm-bound",
    [matcher = std::regex(s)](DeviceSpec const& spec) -> bool {
      return std::regex_match(spec.name, matcher);
    },
    [maxSharedMemory](ComputingQuotaOffer const& offer) -> int8_t { return offer.sharedMemory >= maxSharedMemory ? 127 : 0; }};
}

} // namespace o2::framework

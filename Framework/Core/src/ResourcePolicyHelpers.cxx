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
    [](ComputingQuotaOffer const&, ComputingQuotaOffer const&) -> OfferScore { return OfferScore::Enough; }};
}

ResourcePolicy ResourcePolicyHelpers::cpuBoundTask(char const* s, int requestedCPUs)
{
  return ResourcePolicy{
    "cpu-bound",
    [matcher = std::regex(s)](DeviceSpec const& spec) -> bool {
      return std::regex_match(spec.name, matcher);
    },
    [requestedCPUs](ComputingQuotaOffer const& offer, ComputingQuotaOffer const& accumulated) -> OfferScore { return accumulated.cpu >= requestedCPUs ? OfferScore::Enough : OfferScore::More; }};
}

ResourcePolicy ResourcePolicyHelpers::sharedMemoryBoundTask(char const* s, int requestedSharedMemory)
{
  return ResourcePolicy{
    "shm-bound",
    [matcher = std::regex(s)](DeviceSpec const& spec) -> bool {
      return std::regex_match(spec.name, matcher);
    },
    [requestedSharedMemory](ComputingQuotaOffer const& offer, ComputingQuotaOffer const& accumulated) -> OfferScore { 
      if (offer.sharedMemory == 0) {
        return OfferScore::Unneeded;
      }
      return accumulated.sharedMemory >= requestedSharedMemory ? OfferScore::Enough : OfferScore::More; }};
}

} // namespace o2::framework

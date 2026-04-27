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

#include "Framework/ResourcePolicyHelpers.h"
#include "Framework/DeviceSpec.h"

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
    [requestedCPUs](ComputingQuotaOffer const& offer, ComputingQuotaOffer const& accumulated) -> OfferScore { return accumulated.cpu >= requestedCPUs ? OfferScore::Enough : OfferScore::More; },
    ComputingQuotaOffer{.cpu = requestedCPUs}};
}

ResourcePolicy ResourcePolicyHelpers::rateLimitedSharedMemoryBoundTask(char const* s, int requestedSharedMemory, int requestedTimeslices)
{
  return ResourcePolicy{
    "ratelimited-shm-bound",
    [matcher = std::regex(s)](DeviceSpec const& spec) -> bool {
      return std::regex_match(spec.name, matcher);
    },
    [requestedSharedMemory, requestedTimeslices](ComputingQuotaOffer const& offer, ComputingQuotaOffer const& accumulated) -> OfferScore {
      // If we have enough memory and not enough timeslices,
      // ignore further shared memory.
      if (accumulated.sharedMemory >= requestedSharedMemory && offer.timeslices == 0) {
        return OfferScore::Unneeded;
      }
      // If we have enough timeslices and not enough shared memory
      // ignore further timeslices.
      if (accumulated.timeslices >= requestedTimeslices && offer.sharedMemory == 0) {
        return OfferScore::Unneeded;
      }
      // If it does not offer neither shared memory nor timeslices, mark it as unneeded.
      if (offer.sharedMemory == 0 && offer.timeslices == 0) {
        return OfferScore::Unneeded;
      }
      // We have enough to process.
      if ((accumulated.sharedMemory + offer.sharedMemory) >= requestedSharedMemory && (accumulated.timeslices + offer.timeslices) >= requestedTimeslices) {
        return OfferScore::Enough;
      }
      // We need more resources
      return OfferScore::More; },
    ComputingQuotaOffer{.sharedMemory = requestedSharedMemory, .timeslices = requestedTimeslices}};
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
      return (accumulated.sharedMemory + offer.sharedMemory) >= requestedSharedMemory ? OfferScore::Enough : OfferScore::More; },
    ComputingQuotaOffer{.sharedMemory = requestedSharedMemory}};
}

} // namespace o2::framework

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_RESOURCEPOLICY_H_
#define O2_FRAMEWORK_RESOURCEPOLICY_H_

#include "Framework/ComputingQuotaOffer.h"
#include <functional>
#include <string>

namespace o2::framework
{
struct DeviceSpec;

/// A policy which specify how a device matched by
/// @a matcher should react to a given offer by specifying
/// a given @a request.
struct ResourcePolicy {
  using Matcher = std::function<bool(DeviceSpec const& device)>;

  static std::vector<ResourcePolicy> createDefaultPolicies();

  std::string name;
  Matcher matcher;
  ComputingQuotaRequest request;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_RESOURCEPOLICY_H_

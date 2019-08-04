// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/InputRecord.h"
#include <functional>
#include <iostream>

namespace o2
{
namespace framework
{

/// By default the CompletionPolicy matches any Device and only runs a
/// computation when all the inputs are there.
std::vector<CompletionPolicy>
  CompletionPolicy::createDefaultPolicies()
{
  return {CompletionPolicyHelpers::consumeWhenAll()};
}

std::ostream& operator<<(std::ostream& oss, CompletionPolicy::CompletionOp const& val)
{
  switch (val) {
    case CompletionPolicy::CompletionOp::Consume:
      oss << "consume";
      break;
    case CompletionPolicy::CompletionOp::Process:
      oss << "process";
      break;
    case CompletionPolicy::CompletionOp::Wait:
      oss << "wait";
      break;
    case CompletionPolicy::CompletionOp::Discard:
      oss << "discard";
      break;
  };
  return oss;
}

} // namespace framework
} // namespace o2

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

#include "Framework/O2DataModelHelpers.h"
#include "Framework/DataSpecUtils.h"

namespace o2::framework
{
void O2DataModelHelpers::updateMissingSporadic(fair::mq::Parts& parts, std::vector<OutputSpec> const& specs, std::vector<bool>& present)
{
  // Mark as present anything which is not of Lifetime timeframe.
  for (size_t i = 0; i < specs.size(); ++i) {
    if (specs[i].lifetime != Lifetime::Timeframe) {
      present[i] = true;
    }
  }

  auto timeframeDataExists = [&present, &specs](auto const& dh) -> void {
    for (size_t i = 0; i < specs.size(); ++i) {
      // We already found this, no need to check again.
      if (present[i]) {
        continue;
      }
      // The header is not there, we do not care.
      if (dh == nullptr) {
        continue;
      }
      // The header matcher this output, we mark it as present.
      if (DataSpecUtils::match(specs[i], ConcreteDataMatcher{dh->dataOrigin, dh->dataDescription, dh->subSpecification})) {
        present[i] = true;
      }
    }
  };
  O2DataModelHelpers::for_each_header(parts, timeframeDataExists);
}

std::string O2DataModelHelpers::describeMissingOutputs(std::vector<OutputSpec> const& specs, std::vector<bool> const& present)
{
  assert(specs.size() == present.size());
  std::string error = "This timeframe has a missing output of lifetime timeframe: ";
  bool first = true;
  for (size_t i = 0; i < specs.size(); ++i) {
    if (present[i] == false) {
      if (first) {
        first = false;
      } else {
        error += ", ";
      }
      error += DataSpecUtils::describe(specs[i]);
    }
  }
  error += ". If this is expected, please change its lifetime to Sporadic / QA.";
  first = true;
  for (size_t i = 0; i < specs.size(); ++i) {
    if (present[i] == true) {
      if (first) {
        error += " Present outputs are: ";
        first = false;
      } else {
        error += ", ";
      }
      error += DataSpecUtils::describe(specs[i]);
    }
  }
  if (first) {
    error += " No output was present.";
  } else {
    error += ".";
  }
  return error;
}
} // namespace o2::framework

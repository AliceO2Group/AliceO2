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

#include "Framework/DataProcessorSpecHelpers.h"
#include "Framework/Logger.h"
#include <string>
#include <algorithm>

namespace o2::framework
{
bool DataProcessorSpecHelpers::hasLabel(DataProcessorSpec const& spec, char const* label)
{
  auto sameLabel = [other = DataProcessorLabel{{label}}](DataProcessorLabel const& label) { return label == other; };
  return std::find_if(spec.labels.begin(), spec.labels.end(), sameLabel) != spec.labels.end();
}

DataProcessorSpec DataProcessorSpecHelpers::placeholder(std::string name)
{
  return DataProcessorSpec{
    .name = name,
    .algorithm = AlgorithmSpec::InitCallback{[name](InitContext&) -> AlgorithmSpec::ProcessCallback {
      LOGP(fatal, "{} was not built in this process, which therefore cannot run it.", name);
      return {};
    }},
    .metadata = {{"dpl-placeholder", "true"}}};
}

bool DataProcessorSpecHelpers::isPlaceholder(DataProcessorSpec const& spec)
{
  return std::ranges::any_of(spec.metadata, [](auto const& m) { return m.key == "dpl-placeholder"; });
}
} // namespace o2::framework

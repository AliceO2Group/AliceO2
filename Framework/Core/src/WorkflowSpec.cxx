// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataDescriptorQueryBuilder.h"

#include <cstddef>
#include <functional>
#include <string>

namespace o2
{
namespace framework
{

WorkflowSpec parallel(DataProcessorSpec original,
                      size_t maxIndex,
                      std::function<void(DataProcessorSpec&, size_t)> amendCallback) {
  WorkflowSpec results;
  results.reserve(maxIndex);
  for (size_t i = 0; i < maxIndex; ++i) {
    results.push_back(original);
    results.back().name = original.name + "_" + std::to_string(i);
    results.back().rank = i;
    results.back().nSlots = maxIndex;
    amendCallback(results.back(), i);
  }
  return results;
}

WorkflowSpec parallel(WorkflowSpec specs,
                      size_t maxIndex,
                      std::function<void(DataProcessorSpec&, size_t)> amendCallback)
{
  WorkflowSpec results;
  results.reserve(specs.size() * maxIndex);
  for (auto& spec : specs) {
    auto result = parallel(spec, maxIndex, amendCallback);
    results.insert(results.end(), result.begin(), result.end());
  }

  return results;
}

Inputs mergeInputs(InputSpec original,
                   size_t maxIndex,
                   std::function<void(InputSpec &, size_t)> amendCallback) {
  Inputs results;
  results.reserve(maxIndex);
  for (size_t i = 0; i < maxIndex; ++i) {
    results.push_back(original);
    amendCallback(results.back(), i);
  }
  return results;
}

Inputs mergeInputs(Inputs inputs,
                   size_t maxIndex,
                   std::function<void(InputSpec&, size_t)> amendCallback)
{
  Inputs results;
  results.reserve(inputs.size() * maxIndex);
  for (size_t i = 0; i < maxIndex; ++i) {
    for (auto const& original : inputs) {
      results.push_back(original);
      amendCallback(results.back(), i);
    }
  }
  return results;
}

DataProcessorSpec timePipeline(DataProcessorSpec original,
                          size_t count) {
  if (original.maxInputTimeslices != 1) {
    std::runtime_error("You can time slice only once");
  }
  original.maxInputTimeslices = count;
  return original;
}

/// Really a wrapper around `DataDescriptorQueryBuilder::parse`
/// FIXME: should really use an rvalue..
std::vector<InputSpec> select(const char* matcher)
{
  return DataDescriptorQueryBuilder::parse(matcher);
}

} // namespace framework
} // namespace o2

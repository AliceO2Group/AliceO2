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
#include "Framework/DataSpecUtils.h"

#include <cstddef>
#include <functional>
#include <string>

namespace o2
{
namespace framework
{

WorkflowSpec parallel(DataProcessorSpec original,
                      size_t maxIndex,
                      std::function<void(DataProcessorSpec&, size_t)> amendCallback)
{
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

WorkflowSpec parallelPipeline(const WorkflowSpec& specs,
                              size_t nPipelines,
                              std::function<size_t()> getNumberOfSubspecs,
                              std::function<size_t(size_t)> getSubSpec)
{
  WorkflowSpec result;
  size_t numberOfSubspecs = getNumberOfSubspecs();
  if (numberOfSubspecs < nPipelines) {
    // no need to create more pipelines than the number of parallel Ids, in that case
    // each pipeline serves one id
    nPipelines = numberOfSubspecs;
  }
  for (auto process : specs) {
    size_t index = 0;
    size_t inputMultiplicity = numberOfSubspecs / nPipelines;
    if (numberOfSubspecs % nPipelines) {
      inputMultiplicity += 1;
    }
    auto amendProcess = [numberOfSubspecs, nPipelines, &index, &inputMultiplicity, getSubSpec](DataProcessorSpec& spec, size_t pipeline) {
      auto inputs = std::move(spec.inputs);
      auto outputs = std::move(spec.outputs);
      spec.inputs.reserve(inputMultiplicity);
      spec.outputs.reserve(inputMultiplicity);
      for (size_t inputNo = 0; inputNo < inputMultiplicity; ++inputNo) {
        for (auto& input : inputs) {
          spec.inputs.push_back(input);
          spec.inputs.back().binding += std::to_string(inputNo);
          DataSpecUtils::updateMatchingSubspec(spec.inputs.back(), getSubSpec(index + inputNo));
        }
        for (auto& output : outputs) {
          spec.outputs.push_back(output);
          spec.outputs.back().binding.value += std::to_string(inputNo);
          // FIXME: this will be unneeded once we have a subSpec-less variant...
          DataSpecUtils::updateMatchingSubspec(spec.outputs.back(), getSubSpec(index + inputNo));
        }
      }
      index += inputMultiplicity;
      if (inputMultiplicity > numberOfSubspecs / nPipelines &&
          ((numberOfSubspecs - index) % (nPipelines - (pipeline + 1))) == 0) {
        // if the remaining ids can be distributed equally among the remaining pipelines
        // we can decrease multiplicity
        inputMultiplicity = numberOfSubspecs / nPipelines;
      }
    };

    if (nPipelines > 1) {
      // add multiple processes and distribute inputs among them
      auto amendedProcessors = parallel(process, nPipelines, amendProcess);
      result.insert(result.end(), amendedProcessors.begin(), amendedProcessors.end());
    } else if (nPipelines == 1) {
      // add one single process with all the inputs
      amendProcess(process, 0);
      result.push_back(process);
    }
  }
  return result;
}

Inputs mergeInputs(InputSpec original,
                   size_t maxIndex,
                   std::function<void(InputSpec&, size_t)> amendCallback)
{
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
                               size_t count)
{
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

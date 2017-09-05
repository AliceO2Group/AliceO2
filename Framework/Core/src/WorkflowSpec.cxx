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
#include <functional>
#include <string>
#include <cstddef>

namespace o2 {
namespace framework {

WorkflowSpec parallel(DataProcessorSpec original,
                      size_t maxIndex,
                      std::function<void(DataProcessorSpec&, size_t)> amendCallback) {
  WorkflowSpec results;
  for (size_t i = 0; i < maxIndex; ++i) {
    results.push_back(original);
    results.back().name = original.name + "_" + std::to_string(i);
    results.back().rank = i;
    results.back().nSlots = maxIndex;
    amendCallback(results.back(), i);
  }
  return results;
}

Inputs mergeInputs(InputSpec original,
                   size_t maxIndex,
                   std::function<void(InputSpec &, size_t)> amendCallback) {
  Inputs results;
  for (size_t i = 0; i < maxIndex; ++i) {
    results.push_back(original);
    amendCallback(results.back(), i);
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


} // namespace framework
} // namespace o2

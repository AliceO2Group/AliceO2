// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_RUN_DATA_PROCESSING_H
#define FRAMEWORK_RUN_DATA_PROCESSING_H

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include <vector>
#include <unistd.h>

namespace o2 {
namespace framework {
using WorkflowSpec = std::vector<DataProcessorSpec>;
using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;

}
}

/// To be implemented by the user to specify one or more DataProcessorSpec.
/// The reason why this passes a preallocated specs, rather than asking the
/// caller to allocate his / her own is that if we end up wrapping this in
/// some scripting language, we do not need to delegate the allocation to the
/// scripting language itself.
void  defineDataProcessing(o2::framework::WorkflowSpec &specs);

// This comes from the framework itself. This way we avoid code duplication.
int doMain(int argc, char **argv, const o2::framework::WorkflowSpec &specs);

int main(int argc, char**argv) {
  o2::framework::WorkflowSpec specs;
  defineDataProcessing(specs);
  auto result = doMain(argc, argv, specs);
  std::cout << "Process " << getpid() << " is exiting." << std::endl;
  return result;
}

#endif

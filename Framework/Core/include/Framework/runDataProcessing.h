#ifndef FRAMEWORK_RUN_DATA_PROCESSING_H
#define FRAMEWORK_RUN_DATA_PROCESSING_H

#include "Framework/DataProcessorSpec.h"
#include <vector>
#include <unistd.h>

namespace o2 {
namespace framework {
using WorkflowSpec = std::vector<DataProcessorSpec>;
}
}

// to be implemented by the user to specify one or more DataProcessorSpec
void defineDataProcessing(std::vector<o2::framework::DataProcessorSpec> &specs);

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

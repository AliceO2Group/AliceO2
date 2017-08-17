// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSOR_SPEC_H
#define FRAMEWORK_DATAPROCESSOR_SPEC_H

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/DataRef.h"
#include "Framework/DataAllocator.h"

#include <vector>
#include <string>

namespace o2 {
namespace framework {

class ConfigParamRegistry;
class ServiceRegistry;

struct DataProcessorSpec {
  using InitCallback = std::function<void(const ConfigParamRegistry &, ServiceRegistry &)>;
  using ProcessCallback = std::function<void(const std::vector<DataRef>, ServiceRegistry&, DataAllocator&)>;
  using ErrorCallback = std::function<void(const std::vector<DataRef>, ServiceRegistry &, std::exception &e)>;

  std::string name;
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;
  ProcessCallback process;

  std::vector<ConfigParamSpec> configParams;
  std::vector<std::string> requiredServices;
  InitCallback init;
  ErrorCallback onError;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATAPROCESSOR_SPEC_H

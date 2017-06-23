// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSORSPEC_H
#define FRAMEWORK_DATAPROCESSORSPEC_H

#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/DataRef.h"
#include "Framework/DataAllocator.h"
#include "Framework/AlgorithmSpec.h"

#include <vector>
#include <string>

namespace o2 {
namespace framework {

using Inputs = std::vector<InputSpec>;
using Outputs = std::vector<OutputSpec>;
using Options = std::vector<ConfigParamSpec>;

class ConfigParamRegistry;
class ServiceRegistry;

struct DataProcessorSpec {
  std::string name;
  Inputs inputs;
  Outputs outputs;
  AlgorithmSpec algorithm;

  Options options;
  // FIXME: not used for now...
  std::vector<std::string> requiredServices;
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_DATAPROCESSORSPEC_H

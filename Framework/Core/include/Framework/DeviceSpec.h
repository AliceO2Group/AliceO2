// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DEVICESPEC_H
#define FRAMEWORK_DEVICESPEC_H

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ChannelSpec.h"
#include "Framework/DeviceControl.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include <vector>
#include <string>
#include <map>

namespace o2 {
namespace framework {

/// Concrete description of the device which will actually run 
/// a DataProcessor.
struct DeviceSpec {
  std::string id;
  std::vector<ChannelSpec> channels;
  std::vector<std::string> arguments;
  std::vector<ConfigParamSpec> options;

  AlgorithmSpec algorithm;

  std::map<std::string, InputSpec> inputs;
  std::map<std::string, OutputSpec> outputs;
  std::map<std::string, InputSpec> forwards;
  std::vector<char *> args; // Calculated list of args for the device.
  size_t rank;
  size_t nSlots;
};

/// Helper to convert from an abstract dataflow specification, @a workflow,
/// to an actual set of devices which will have to run.
void
dataProcessorSpecs2DeviceSpecs(const o2::framework::WorkflowSpec &workflow,
                               std::vector<o2::framework::DeviceSpec> &devices);

/// Helper to prepare the arguments which will be used to 
/// start the various devices.
void
prepareArguments(int argc,
                 char **argv,
                 bool defaultQuiet,
                 bool defaultStopped,
                 std::vector<DeviceSpec> &deviceSpecs,
                 std::vector<DeviceControl> &deviceControls);
}
}
#endif

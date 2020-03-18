// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DeviceSpecHelpers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceExecution.h"
#include "../src/DeviceSpecHelpers.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/detail/per_element_manip.hpp>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>
#include "../src/SimpleResourceManager.h"
#include "../src/ComputingResourceHelpers.h"

namespace o2
{
namespace framework
{

using CheckMatrix = std::map<std::string, std::vector<std::pair<std::string, std::string>>>;

// search for an option in the device execution
bool search(DeviceExecution const& execution, std::string const& option, std::string const& argument)
{
  bool foundOption = false;
  for (auto const& execArg : execution.args) {
    if (execArg == nullptr) {
      break;
    }
    if (!foundOption) {
      foundOption = option == execArg;
    } else if (argument == execArg) {
      return true;
    } else {
      // the required argument to the option is not found
      foundOption = false;
    }
  }
  return false;
}

// create the device execution from the device specs and process the command line arguments
// the check matrix contains a map of options to be founf per processor
void check(const std::vector<std::string>& arguments,
           const std::vector<ConfigParamSpec>& workflowOptions,
           const std::vector<DeviceSpec>& deviceSpecs,
           CheckMatrix& matrix)
{
  std::stringstream output;
  for (auto const& arg : arguments) {
    output << " " << arg;
  }
  std::cout << "checking for arguments: " << output.str() << std::endl;

  std::vector<DeviceExecution> deviceExecutions(deviceSpecs.size());
  std::vector<DeviceControl> deviceControls(deviceSpecs.size());
  std::vector<DataProcessorInfo> dataProcessorInfos;
  for (auto& [name, _] : matrix) {
    dataProcessorInfos.push_back(DataProcessorInfo{
      name,
      "executable-name",
      arguments,
      workflowOptions,
    });
  }
  DeviceSpecHelpers::prepareArguments(true, true,
                                      dataProcessorInfos,
                                      deviceSpecs,
                                      deviceExecutions,
                                      deviceControls,
                                      "workflow-id");

  std::cout << "created execution for " << deviceSpecs.size() << " device(s)" << std::endl;

  for (size_t index = 0; index < deviceSpecs.size(); index++) {
    const auto& deviceSpec = deviceSpecs[index];
    const auto& deviceExecution = deviceExecutions[index];
    std::cout << deviceSpec.name << std::endl;
    std::stringstream execArgs;
    for (const auto& arg : deviceExecution.args) {
      if (arg == nullptr) {
        // the nullptr terminates the argument list
        break;
      }
      execArgs << "  " << arg;
    }
    std::cout << execArgs.str() << std::endl;
    for (auto const& testCase : matrix[deviceSpec.name]) {
      BOOST_TEST_INFO(std::string("can not find option: ") + testCase.first + " " + testCase.second);
      BOOST_CHECK(search(deviceExecution, testCase.first, testCase.second));
    }
  }
}

BOOST_AUTO_TEST_CASE(test_prepareArguments)
{
  std::vector<ConfigParamSpec> workflowOptions{
    {"foo", VariantType::String, "bar", {"the famous foo option"}},
    {"depth", VariantType::Int, 1, {"number of processors"}},
  };

  auto algorithm = [](ProcessingContext& ctx) {};

  WorkflowSpec workflow{
    {"processor0",
     {},
     {OutputSpec{{"output"}, "TST", "DUMMYDATA", 0, Lifetime::Timeframe}},
     AlgorithmSpec(algorithm),
     Options{
       {"mode", VariantType::String, "default", {"The Mode"}},
     }},
    {"processor1",
     {InputSpec{"input", "TST", "DUMMYDATA", 0, Lifetime::Timeframe}},
     {},
     AlgorithmSpec(algorithm),
     Options{
       {"mode", VariantType::String, "default", {"The Mode"}},
     }},
  };

  std::vector<DeviceSpec> deviceSpecs;

  std::vector<ComputingResource> resources = {ComputingResourceHelpers::getLocalhostResource()};
  auto rm = std::make_unique<SimpleResourceManager>(resources);

  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow,
                                                    ChannelConfigurationPolicy::createDefaultPolicies(),
                                                    CompletionPolicy::createDefaultPolicies(),
                                                    deviceSpecs,
                                                    *rm, "workflow-id");

  // Now doing the test cases
  CheckMatrix matrix;

  // checking with empty command line arguments, all processors must have the options with
  // default arguments
  matrix["processor0"] = {{"--depth", "1"}, {"--foo", "bar"}, {"--mode", "default"}};
  matrix["processor1"] = matrix["processor0"];
  check({}, workflowOptions, deviceSpecs, matrix);

  // checking with unknown arguments, silently ignored, same test matrix
  check({"--unknown", "option"}, workflowOptions, deviceSpecs, matrix);

  // configuring mode, both devices must have the option set
  matrix["processor0"] = {{"--depth", "1"}, {"--foo", "bar"}, {"--mode", "silly"}};
  matrix["processor1"] = matrix["processor0"];
  check({"--mode", "silly"}, workflowOptions, deviceSpecs, matrix);

  // configuring option group, only processor0 must have the option set, processor1 default
  matrix["processor0"] = {{"--depth", "1"}, {"--foo", "bar"}, {"--mode", "silly"}};
  matrix["processor1"] = {{"--depth", "1"}, {"--foo", "bar"}, {"--mode", "default"}};
  check({"--processor0", "--mode silly"}, workflowOptions, deviceSpecs, matrix);

  // processor0 must have the mode set to silly via option group, processor1 advanced from the argument
  matrix["processor0"] = {{"--depth", "1"}, {"--foo", "bar"}, {"--mode", "silly"}};
  matrix["processor1"] = {{"--depth", "1"}, {"--foo", "bar"}, {"--mode", "advanced"}};
  check({"--mode", "advanced", "--processor0", "--mode silly"}, workflowOptions, deviceSpecs, matrix);

  // both devices have the workflow option propagated, others defaulted
  matrix["processor0"] = {{"--depth", "2"}, {"--foo", "bar"}, {"--mode", "default"}};
  matrix["processor1"] = matrix["processor0"];
  check({"--depth", "2"}, workflowOptions, deviceSpecs, matrix);

  // both devices have the workflow option propagated, processor0 mode silly via option group
  matrix["processor0"] = {{"--depth", "2"}, {"--foo", "bar"}, {"--mode", "silly"}};
  matrix["processor1"] = {{"--depth", "2"}, {"--foo", "bar"}, {"--mode", "default"}};
  check({"--depth", "2", "--processor0", "--mode silly"}, workflowOptions, deviceSpecs, matrix);
}
} // namespace framework
} // namespace o2

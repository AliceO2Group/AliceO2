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

#include "Mocking.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DeviceExecution.h"
#include "Framework/DriverConfig.h"
#include "../src/DeviceSpecHelpers.h"
#include <catch_amalgamated.hpp>
#include <algorithm>
#include <sstream>
#include <cstring>
#include <vector>
#include <map>
#include <cstring>
#include "../src/SimpleResourceManager.h"
#include "../src/ComputingResourceHelpers.h"

namespace o2::framework
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
  DriverConfig driverConfig{};
  DeviceSpecHelpers::prepareArguments(true, true, false, 8080,
                                      driverConfig,
                                      dataProcessorInfos,
                                      deviceSpecs,
                                      deviceExecutions,
                                      deviceControls,
                                      "workflow-id");

  for (size_t index = 0; index < deviceSpecs.size(); index++) {
    const auto& deviceSpec = deviceSpecs[index];
    const auto& deviceExecution = deviceExecutions[index];
    std::stringstream execArgs;
    for (const auto& arg : deviceExecution.args) {
      if (arg == nullptr) {
        // the nullptr terminates the argument list
        break;
      }
      execArgs << "  " << arg;
    }
    for (auto const& testCase : matrix[deviceSpec.name]) {
      INFO(std::string("can not find option: ") + testCase.first + " " + testCase.second);
      REQUIRE(search(deviceExecution, testCase.first, testCase.second));
    }
  }
}

TEST_CASE("test_prepareArguments")
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

  auto configContext = makeEmptyConfigContext();
  auto channelPolicies = ChannelConfigurationPolicy::createDefaultPolicies(*configContext);
  auto callbacksPolicies = CallbacksPolicy::createDefaultPolicies();

  DeviceSpecHelpers::dataProcessorSpecs2DeviceSpecs(workflow,
                                                    channelPolicies,
                                                    CompletionPolicy::createDefaultPolicies(),
                                                    callbacksPolicies,
                                                    deviceSpecs,
                                                    *rm, "workflow-id", *configContext);

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

TEST_CASE("CheckOptionReworking")
{
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"--driver-client-backend", "foo"}},
      {}};
    DeviceSpecHelpers::reworkHomogeneousOption(infos, "--driver-client-backend", "stdout://");
    REQUIRE(infos[0].cmdLineArgs[1] == "foo");
    REQUIRE(infos[1].cmdLineArgs[1] == "foo");
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"--driver-client-backend", "foo"}},
      {{}, {}, {"--driver-client-backend", "bar"}}};
    REQUIRE_THROWS_AS(DeviceSpecHelpers::reworkHomogeneousOption(infos, "--driver-client-backend", "stdout://"), o2::framework::RuntimeErrorRef);
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"--driver-client-backend", "foo"}},
      {{}, {}, {"--driver-client-backend", "foo"}}};
    DeviceSpecHelpers::reworkHomogeneousOption(infos, "--driver-client-backend", "stdout://");
    REQUIRE(infos[0].cmdLineArgs[1] == "foo");
    REQUIRE(infos[1].cmdLineArgs[1] == "foo");
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"foo", "bar"}},
      {{}, {}, {"fnjcnak", "foo"}}};
    DeviceSpecHelpers::reworkHomogeneousOption(infos, "--driver-client-backend", "stdout://");
    REQUIRE(infos[0].cmdLineArgs[3] == "stdout://");
    REQUIRE(infos[1].cmdLineArgs[3] == "stdout://");
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"foo", "bar", "--driver-client-backend", "bar"}},
      {{}, {}, {"fnjcnak", "foo"}}};
    DeviceSpecHelpers::reworkHomogeneousOption(infos, "--driver-client-backend", "stdout://");
    REQUIRE(infos[0].cmdLineArgs[3] == "bar");
    REQUIRE(infos[1].cmdLineArgs[3] == "bar");
  }
}

TEST_CASE("CheckIntegerReworking")
{
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"--readers", "2"}},
      {}};
    DeviceSpecHelpers::reworkIntegerOption(
      infos, "--readers", nullptr, 1, [](long long x, long long y) { return x > y ? x : y; });
    REQUIRE(infos[0].cmdLineArgs[1] == "2");
    REQUIRE(infos[1].cmdLineArgs[1] == "2");
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {},
      {}};
    DeviceSpecHelpers::reworkIntegerOption(
      infos, "--readers", nullptr, 1, [](long long x, long long y) { return x > y ? x : y; });
    REQUIRE(infos[0].cmdLineArgs.size() == 0);
    REQUIRE(infos[1].cmdLineArgs.size() == 0);
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {},
      {}};
    DeviceSpecHelpers::reworkIntegerOption(
      infos, "--readers", []() { return 1; }, 3, [](long long x, long long y) { return x > y ? x : y; });
    REQUIRE(infos[0].cmdLineArgs.size() == 2);
    REQUIRE(infos[1].cmdLineArgs.size() == 2);
    REQUIRE(infos[0].cmdLineArgs[1] == "1");
    REQUIRE(infos[1].cmdLineArgs[1] == "1");
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"--readers", "2"}},
      {{}, {}, {"--readers", "3"}}};
    DeviceSpecHelpers::reworkIntegerOption(
      infos, "--readers", []() { return 1; }, 1, [](long long x, long long y) { return x > y ? x : y; });
    REQUIRE(infos[0].cmdLineArgs.size() == 2);
    REQUIRE(infos[1].cmdLineArgs.size() == 2);
    REQUIRE(infos[0].cmdLineArgs[1] == "3");
    REQUIRE(infos[1].cmdLineArgs[1] == "3");
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"--readers", "3"}},
      {{}, {}, {"--readers", "2"}}};
    DeviceSpecHelpers::reworkIntegerOption(
      infos, "--readers", []() { return 1; }, 1, [](long long x, long long y) { return x > y ? x : y; });
    REQUIRE(infos[0].cmdLineArgs.size() == 2);
    REQUIRE(infos[1].cmdLineArgs.size() == 2);
    REQUIRE(infos[0].cmdLineArgs[1] == "3");
    REQUIRE(infos[1].cmdLineArgs[1] == "3");
  }
  {
    std::vector<DataProcessorInfo> infos = {
      {{}, {}, {"foo", "bar", "--readers", "3"}},
      {{}, {}, {"--readers", "2"}}};
    DeviceSpecHelpers::reworkIntegerOption(
      infos, "--readers", []() { return 1; }, 1, [](long long x, long long y) { return x > y ? x : y; });
    REQUIRE(infos[0].cmdLineArgs.size() == 4);
    REQUIRE(infos[1].cmdLineArgs.size() == 2);
    REQUIRE(infos[0].cmdLineArgs[3] == "3");
    REQUIRE(infos[1].cmdLineArgs[1] == "3");
  }
}

TEST_CASE("Check validity of the workflow")
{
  WorkflowSpec workflow1{
    {
      .name = "processor0",
      .outputs = {OutputSpec{{"output"}, "TST", "DUMMYDATA", 0, Lifetime::Timeframe}},
    },
    {
      .name = "processor1",
      .inputs = {InputSpec{"input", "TST", "DUMMYDATA", 0, Lifetime::Timeframe}},
    },
  };
  REQUIRE_NOTHROW(DeviceSpecHelpers::validate(workflow1));

  WorkflowSpec workflow2{
    {
      .name = "processor0",
      .outputs = {OutputSpec{{"output"}, "TST", "DUMMYDATA", 0, Lifetime::Timeframe},
                  OutputSpec{{"output2"}, "TST", "DUMMYDATA", 0, Lifetime::Timeframe}},
    },
    {
      .name = "processor1",
      .inputs = {InputSpec{"input", "TST", "DUMMYDATA", 0, Lifetime::Timeframe}},
    },
  };
  REQUIRE_THROWS(DeviceSpecHelpers::validate(workflow2));

  WorkflowSpec workflow3{
    {
      .name = "processor0",
      .outputs = {OutputSpec{{"output"}, "TST", "DUMMYDATA", 0, Lifetime::Timeframe},
                  OutputSpec{{"output2"}, "TST", "DUMMYDATA", 2, Lifetime::Sporadic}},
    },
    {
      .name = "processor1",
      .outputs = {OutputSpec{{"output"}, "TST", "DUMMYDATA", 1, Lifetime::Timeframe},
                  OutputSpec{{"output2"}, "TST", "DUMMYDATA", 2, Lifetime::Sporadic}},
    },
    {
      .name = "processor2",
      .inputs = {InputSpec{"input", "TST", "DUMMYDATA", 2, Lifetime::Timeframe}},
    },
  };
  REQUIRE_NOTHROW(DeviceSpecHelpers::validate(workflow3));

  WorkflowSpec workflow4{
    {
      .name = "processor0",
      .outputs = {OutputSpec{{"output"}, "TST", "DUMMYDATA", 0, Lifetime::Timeframe},
                  OutputSpec{{"output2"}, "TST", "DUMMYDATA", 2, Lifetime::Sporadic}},
    },
    {
      .name = "processor1",
      .outputs = {OutputSpec{{"output"}, "TST", "DUMMYDATA", 0, Lifetime::Timeframe},
                  OutputSpec{{"output2"}, "TST", "DUMMYDATA", 2, Lifetime::Sporadic}},
    },
    {
      .name = "processor2",
      .inputs = {InputSpec{"input", "TST", "DUMMYDATA", 2, Lifetime::Timeframe}},
    },
  };
  REQUIRE_THROWS(DeviceSpecHelpers::validate(workflow4));
}

TEST_CASE("CheckReworkingEnv")
{
  DeviceSpec spec{.inputTimesliceId = 1};
  std::string env = "FOO={timeslice0}";
  REQUIRE(DeviceSpecHelpers::reworkEnv(env, spec) == "FOO=1");
  env = "FOO={timeslice0} BAR={timeslice4}";
  REQUIRE(DeviceSpecHelpers::reworkEnv(env, spec) == "FOO=1 BAR=5");
  env = "FOO={timeslice0} BAR={timeslice4} BAZ={timeslice5}";
  REQUIRE(DeviceSpecHelpers::reworkEnv(env, spec) == "FOO=1 BAR=5 BAZ=6");
  env = "";
  REQUIRE(DeviceSpecHelpers::reworkEnv(env, spec) == "");
  env = "Plottigat";
  REQUIRE(DeviceSpecHelpers::reworkEnv(env, spec) == "Plottigat");
  env = "{timeslice}";
  REQUIRE(DeviceSpecHelpers::reworkEnv(env, spec) == "{timeslice}");
  env = "{timeslicepluto";
  REQUIRE(DeviceSpecHelpers::reworkEnv(env, spec) == "{timeslicepluto");
}

} // namespace o2::framework

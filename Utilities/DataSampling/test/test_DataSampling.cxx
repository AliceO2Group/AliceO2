// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DataSampling
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "DataSampling/DataSampling.h"
#include "DataSampling/Dispatcher.h"
#include "DataSampling/DataSamplingPolicy.h"
#include "Framework/DataSpecUtils.h"

#include "Headers/DataHeader.h"

#include <Configuration/ConfigurationFactory.h>

using namespace o2::framework;
using namespace o2::utilities;
using namespace o2::configuration;

using DataHeader = o2::header::DataHeader;
using DataOrigin = o2::header::DataOrigin;
using DataDescription = o2::header::DataDescription;

BOOST_AUTO_TEST_CASE(DataSamplingSimpleFlow)
{
  WorkflowSpec workflow{
    {"producer",
     Inputs{},
     Outputs{{"TPC", "CLUSTERS"}}},
    {"processingStage",
     Inputs{{"dataTPC", "TPC", "CLUSTERS"}},
     Outputs{{"TPC", "CLUSTERS_P"}}}};

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSampling.json";
  std::cout << "config file : "
            << "json:/" << configFilePath << std::endl;
  DataSampling::GenerateInfrastructure(workflow, "json:/" + configFilePath);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name.find("Dispatcher") != std::string::npos;
                           });
  BOOST_REQUIRE(disp != workflow.end());

  auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                            [](const InputSpec& in) {
                              return DataSpecUtils::match(in, DataOrigin("TPC"), DataDescription("CLUSTERS"), 0) && in.lifetime == Lifetime::Timeframe;
                            });
  BOOST_CHECK(input != disp->inputs.end());

  input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                       [](const InputSpec& in) {
                         return DataSpecUtils::match(in, DataOrigin("TPC"), DataDescription("CLUSTERS_P"), 0) && in.lifetime == Lifetime::Timeframe;
                       });
  BOOST_CHECK(input != disp->inputs.end());

  auto output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                             [](const OutputSpec& out) {
                               return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters0", 0}) && out.lifetime == Lifetime::Timeframe;
                             });
  BOOST_CHECK(output != disp->outputs.end());

  output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                        [](const OutputSpec& out) {
                          return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters1", 0}) && out.lifetime == Lifetime::Timeframe;
                        });
  BOOST_CHECK(output != disp->outputs.end());

  BOOST_CHECK(disp->algorithm.onInit != nullptr);
}

BOOST_AUTO_TEST_CASE(DataSamplingParallelFlow)
{
  WorkflowSpec workflow{
    {"producer",
     Inputs{},
     {OutputSpec{"TPC", "CLUSTERS", 0},
      OutputSpec{"TPC", "CLUSTERS", 1},
      OutputSpec{"TPC", "CLUSTERS", 2}}}};

  auto processingStages = parallel(
    DataProcessorSpec{
      "processingStage",
      Inputs{{"dataTPC", "TPC", "CLUSTERS"}},
      Outputs{{"TPC", "CLUSTERS_P"}}},
    3,
    [](DataProcessorSpec& spec, size_t index) {
      DataSpecUtils::updateMatchingSubspec(spec.inputs[0], index);
      DataSpecUtils::updateMatchingSubspec(spec.outputs[0], index);
    });

  workflow.insert(std::end(workflow), std::begin(processingStages), std::end(processingStages));

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSampling.json";
  DataSampling::GenerateInfrastructure(workflow, "json:/" + configFilePath);

  for (int i = 0; i < 3; ++i) {
    auto disp = std::find_if(workflow.begin(), workflow.end(),
                             [i](const DataProcessorSpec& d) {
                               return d.name.find("Dispatcher") != std::string::npos;
                             });
    BOOST_REQUIRE(disp != workflow.end());

    auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                              [i](const InputSpec& in) {
                                return DataSpecUtils::match(in, ConcreteDataMatcher{DataOrigin("TPC"), DataDescription("CLUSTERS"), static_cast<DataHeader::SubSpecificationType>(i)}) && in.lifetime == Lifetime::Timeframe;
                              });
    BOOST_CHECK(input != disp->inputs.end());

    input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                         [i](const InputSpec& in) {
                           return DataSpecUtils::match(in, ConcreteDataMatcher{DataOrigin("TPC"), DataDescription("CLUSTERS_P"), static_cast<DataHeader::SubSpecificationType>(i)}) && in.lifetime == Lifetime::Timeframe;
                         });
    BOOST_CHECK(input != disp->inputs.end());

    auto output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                               [](const OutputSpec& out) {
                                 return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters0", 0}) && out.lifetime == Lifetime::Timeframe;
                               });
    BOOST_CHECK(output != disp->outputs.end());

    output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                          [](const OutputSpec& out) {
                            return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters1", 0}) && out.lifetime == Lifetime::Timeframe;
                          });
    BOOST_CHECK(output != disp->outputs.end());

    BOOST_CHECK(disp->algorithm.onInit != nullptr);
  }
}

BOOST_AUTO_TEST_CASE(DataSamplingTimePipelineFlow)
{
  WorkflowSpec workflow{
    {"producer",
     Inputs{},
     {OutputSpec{"TPC", "CLUSTERS", 0, Lifetime::Timeframe}}},
    timePipeline(
      DataProcessorSpec{
        "processingStage",
        Inputs{
          {"dataTPC", "TPC", "CLUSTERS", 0, Lifetime::Timeframe}},
        Outputs{
          {"TPC", "CLUSTERS_P", 0, Lifetime::Timeframe}}},
      3)};

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSampling.json";
  DataSampling::GenerateInfrastructure(workflow, "json:/" + configFilePath, 3);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name.find("Dispatcher") != std::string::npos;
                           });

  BOOST_REQUIRE(disp != workflow.end());
  BOOST_CHECK_EQUAL(disp->inputs.size(), 4);
  BOOST_CHECK_EQUAL(disp->outputs.size(), 3);
  BOOST_CHECK(disp->algorithm.onInit != nullptr);
  BOOST_CHECK_EQUAL(disp->maxInputTimeslices, 3);
}

BOOST_AUTO_TEST_CASE(InputSpecsForPolicy)
{
  std::string configFilePath = "json:/" + std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSampling.json";
  std::vector<InputSpec> inputs = DataSampling::InputSpecsForPolicy(configFilePath, "tpcclusters");

  BOOST_CHECK_EQUAL(inputs.size(), 2);
  BOOST_CHECK(DataSpecUtils::match(inputs[0], ConcreteDataTypeMatcher{"DS", "tpcclusters0"}));
  BOOST_CHECK_EQUAL(inputs[0].binding, "clusters");
  BOOST_CHECK(DataSpecUtils::match(inputs[1], ConcreteDataTypeMatcher{"DS", "tpcclusters1"}));
  BOOST_CHECK_EQUAL(inputs[1].binding, "clusters_p");

  std::unique_ptr<ConfigurationInterface> config = ConfigurationFactory::getConfiguration(configFilePath);
  inputs = DataSampling::InputSpecsForPolicy(config.get(), "tpcclusters");

  BOOST_CHECK_EQUAL(inputs.size(), 2);
}

BOOST_AUTO_TEST_CASE(DataSamplingEmptyConfig)
{
  std::string configFilePath = "json:/" + std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSamplingEmpty.json";

  WorkflowSpec workflow;
  BOOST_CHECK_NO_THROW(DataSampling::GenerateInfrastructure(workflow, configFilePath));
}

BOOST_AUTO_TEST_CASE(DataSamplingOverlappingInputs)
{
  {
    // policy3 includes 1 and 2, so we should have only one inputspec for data, one for timer
    Dispatcher dispatcher("dispatcher", "");
    auto policy1 = std::make_unique<DataSamplingPolicy>("policy1");
    policy1->registerPath({"vcxz", "TST", "AAAA", 0}, {{"erwv"}, "DS", "AAAA"});

    auto policy2 = std::make_unique<DataSamplingPolicy>("policy2");
    policy2->registerPath({"fdsa", "TST", "AAAA", 0}, {{"fdsf"}, "DS", "BBBB"});

    auto policy3 = std::make_unique<DataSamplingPolicy>("policy3");
    policy3->registerPath({"asdf", {"TST", "AAAA"}}, {{"erwv"}, "DS", "CCCC"});

    dispatcher.registerPolicy(std::move(policy1));
    dispatcher.registerPolicy(std::move(policy2));
    dispatcher.registerPolicy(std::move(policy3));

    auto inputs = dispatcher.getInputSpecs();

    BOOST_REQUIRE_EQUAL(inputs.size(), 2);
    BOOST_CHECK_EQUAL(inputs[0], (InputSpec{"asdf", {"TST", "AAAA"}}));
    BOOST_CHECK_EQUAL(inputs[1], (InputSpec{"timer-stats", "DS", "TIMER-dispatcher", 0, Lifetime::Timer}));
  }

  {
    // policy3 includes 1 and 2, so we should have only one inputspec for data, one for timer
    // same as before, but different order of registration
    Dispatcher dispatcher("dispatcher", "");
    auto policy1 = std::make_unique<DataSamplingPolicy>("policy1");
    policy1->registerPath({"vcxz", "TST", "AAAA", 0}, {{"erwv"}, "DS", "AAAA"});

    auto policy2 = std::make_unique<DataSamplingPolicy>("policy2");
    policy2->registerPath({"fdsa", "TST", "AAAA", 0}, {{"fdsf"}, "DS", "BBBB"});

    auto policy3 = std::make_unique<DataSamplingPolicy>("policy3");
    policy3->registerPath({"asdf", {"TST", "AAAA"}}, {{"erwv"}, "DS", "CCCC"});

    dispatcher.registerPolicy(std::move(policy3));
    dispatcher.registerPolicy(std::move(policy1));
    dispatcher.registerPolicy(std::move(policy2));

    auto inputs = dispatcher.getInputSpecs();

    BOOST_REQUIRE_EQUAL(inputs.size(), 2);
    BOOST_CHECK_EQUAL(inputs[0], (InputSpec{"asdf", {"TST", "AAAA"}}));
    BOOST_CHECK_EQUAL(inputs[1], (InputSpec{"timer-stats", "DS", "TIMER-dispatcher", 0, Lifetime::Timer}));
  }

  {
    // three different inputs + timer
    Dispatcher dispatcher("dispatcher", "");
    auto policy1 = std::make_unique<DataSamplingPolicy>("policy1");
    policy1->registerPath({"vcxz", "TST", "AAAA", 0}, {{"erwv"}, "DS", "AAAA"});

    auto policy2 = std::make_unique<DataSamplingPolicy>("policy2");
    policy2->registerPath({"sfsd", "TST", "BBBB", 0}, {{"fdsf"}, "DS", "BBBB"});

    auto policy3 = std::make_unique<DataSamplingPolicy>("policy3");
    policy3->registerPath({"asdf", {"TST", "CCCC"}}, {{"erwv"}, "DS", "CCCC"});

    dispatcher.registerPolicy(std::move(policy1));
    dispatcher.registerPolicy(std::move(policy2));
    dispatcher.registerPolicy(std::move(policy3));

    auto inputs = dispatcher.getInputSpecs();

    BOOST_REQUIRE_EQUAL(inputs.size(), 4);
    BOOST_CHECK_EQUAL(inputs[0], (InputSpec{"vcxz", "TST", "AAAA"}));
    BOOST_CHECK_EQUAL(inputs[1], (InputSpec{"sfsd", "TST", "BBBB"}));
    BOOST_CHECK_EQUAL(inputs[2], (InputSpec{"asdf", {"TST", "CCCC"}}));
    BOOST_CHECK_EQUAL(inputs[3], (InputSpec{"timer-stats", "DS", "TIMER-dispatcher", 0, Lifetime::Timer}));
  }

  {
    // two policies with one common concrete data spec
    Dispatcher dispatcher("dispatcher", "");
    auto policy1 = std::make_unique<DataSamplingPolicy>("policy1");
    policy1->registerPath({"random", "TST", "AAAA", 0}, {{"erwv"}, "DS", "XYZ", 0});

    auto policy2 = std::make_unique<DataSamplingPolicy>("policy2");
    policy2->registerPath({"random0", "TST", "AAAA", 0}, {{"fdsf"}, "DS", "BBBB", 0});
    policy2->registerPath({"random1", "TST", "AAAA", 1}, {{"fdsf"}, "DS", "BBBB", 1});

    dispatcher.registerPolicy(std::move(policy1));
    dispatcher.registerPolicy(std::move(policy2));

    auto inputs = dispatcher.getInputSpecs();

    BOOST_REQUIRE_EQUAL(inputs.size(), 3);
    BOOST_CHECK_EQUAL(inputs[0], (InputSpec{"random0", "TST", "AAAA", 0}));
    BOOST_CHECK_EQUAL(inputs[1], (InputSpec{"random1", "TST", "AAAA", 1}));
    BOOST_CHECK_EQUAL(inputs[2], (InputSpec{"timer-stats", "DS", "TIMER-dispatcher", 0, Lifetime::Timer}));
  }
}

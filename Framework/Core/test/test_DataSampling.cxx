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

#include "Framework/DataSampling.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/DataSamplingReadoutAdapter.h"
#include "Framework/DataSpecUtils.h"

#include "Headers/DataHeader.h"

#include <Configuration/ConfigurationFactory.h>

using namespace o2::framework;
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
                               return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters-0", 0}) && out.lifetime == Lifetime::Timeframe;
                             });
  BOOST_CHECK(output != disp->outputs.end());

  output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                        [](const OutputSpec& out) {
                          return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters-1", 0}) && out.lifetime == Lifetime::Timeframe;
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
                                 return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters-0", 0}) && out.lifetime == Lifetime::Timeframe;
                               });
    BOOST_CHECK(output != disp->outputs.end());

    output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                          [](const OutputSpec& out) {
                            return DataSpecUtils::match(out, ConcreteDataMatcher{"DS", "tpcclusters-1", 0}) && out.lifetime == Lifetime::Timeframe;
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
  BOOST_CHECK_EQUAL(disp->inputs.size(), 3);
  BOOST_CHECK_EQUAL(disp->outputs.size(), 3);
  BOOST_CHECK(disp->algorithm.onInit != nullptr);
  BOOST_CHECK_EQUAL(disp->maxInputTimeslices, 3);
}

BOOST_AUTO_TEST_CASE(DataSamplingFairMq)
{
  WorkflowSpec workflow{
    specifyExternalFairMQDeviceProxy(
      "readout-proxy",
      Outputs{{"TPC", "RAWDATA"}},
      "fake-channel-config",
      dataSamplingReadoutAdapter({"TPC", "RAWDATA"}))};

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSampling.json";
  DataSampling::GenerateInfrastructure(workflow, "json:/" + configFilePath);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name.find("Dispatcher") != std::string::npos;
                           });
  BOOST_REQUIRE(disp != workflow.end());

  auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                            [](const InputSpec& in) {
                              return DataSpecUtils::match(in, ConcreteDataMatcher{DataOrigin("TPC"), DataDescription("RAWDATA"), 0}) && in.lifetime == Lifetime::Timeframe;
                            });
  BOOST_CHECK(input != disp->inputs.end());

  auto channelConfig = std::find_if(disp->options.begin(), disp->options.end(),
                                    [](const ConfigParamSpec& opt) {
                                      return opt.name == "channel-config";
                                    });
  BOOST_REQUIRE(channelConfig != disp->options.end());
}

BOOST_AUTO_TEST_CASE(InputSpecsForPolicy)
{
  std::string configFilePath = "json:/" + std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSampling.json";
  std::vector<InputSpec> inputs = DataSampling::InputSpecsForPolicy(configFilePath, "tpcclusters");

  BOOST_CHECK_EQUAL(inputs.size(), 2);
  BOOST_CHECK(DataSpecUtils::match(inputs[0], ConcreteDataTypeMatcher{"DS", "tpcclusters-0"}));
  BOOST_CHECK_EQUAL(inputs[0].binding, "clusters");
  BOOST_CHECK(DataSpecUtils::match(inputs[1], ConcreteDataTypeMatcher{"DS", "tpcclusters-1"}));
  BOOST_CHECK_EQUAL(inputs[1].binding, "clusters_p");

  std::unique_ptr<ConfigurationInterface> config = ConfigurationFactory::getConfiguration(configFilePath);
  inputs = DataSampling::InputSpecsForPolicy(config.get(), "tpcclusters");

  BOOST_CHECK_EQUAL(inputs.size(), 2);
}

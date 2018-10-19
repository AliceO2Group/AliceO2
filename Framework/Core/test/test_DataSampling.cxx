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
#include <Framework/DataProcessingHeader.h>
#include <Framework/ExternalFairMQDeviceProxy.h>
#include <Framework/DataSamplingReadoutAdapter.h>

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using DataOrigin = o2::header::DataOrigin;
using DataDescription = o2::header::DataDescription;
//using namespace std;

BOOST_AUTO_TEST_CASE(DataSamplingSimpleFlow)
{
  //  LOG(INFO) << (DataDescription("CLUSTERS_P") == DataDescription("CLUSTERS"));
  WorkflowSpec workflow{
    { "producer",
      Inputs{},
      Outputs{ { "TPC", "CLUSTERS" } } },
    { "processingStage",
      Inputs{ { "dataTPC", "TPC", "CLUSTERS" } },
      Outputs{ { "TPC", "CLUSTERS_P" } } }
  };

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSamplingDPL.json";
  std::cout << "config file : "
            << "json:/" << configFilePath << std::endl;
  DataSampling::GenerateInfrastructure(workflow, "json://" + configFilePath);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name.find("Dispatcher") != std::string::npos;
                           });
  BOOST_REQUIRE(disp != workflow.end());

  auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                            [](const InputSpec& in) {
                              return in.origin == DataOrigin("TPC") &&
                                     in.description == DataDescription("CLUSTERS") &&
                                     in.subSpec == 0 &&
                                     in.lifetime == Lifetime::Timeframe;
                            });
  BOOST_CHECK(input != disp->inputs.end());

  input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                       [](const InputSpec& in) {
                         return in.origin == DataOrigin("TPC") &&
                                in.description == DataDescription("CLUSTERS_P") &&
                                in.subSpec == 0 &&
                                in.lifetime == Lifetime::Timeframe;
                       });
  BOOST_CHECK(input != disp->inputs.end());

  auto output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                             [](const OutputSpec& out) {
                               return out.origin == DataOrigin("DS") &&
                                      out.description == DataDescription("tpcclusters-0") &&
                                      out.subSpec == 0 &&
                                      out.lifetime == Lifetime::Timeframe;
                             });
  BOOST_CHECK(output != disp->outputs.end());

  output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                        [](const OutputSpec& out) {
                          return out.origin == DataOrigin("DS") &&
                                 out.description == DataDescription("tpcclusters-1") &&
                                 out.subSpec == 0 &&
                                 out.lifetime == Lifetime::Timeframe;
                        });
  BOOST_CHECK(output != disp->outputs.end());

  BOOST_CHECK(disp->algorithm.onInit != nullptr);
}


BOOST_AUTO_TEST_CASE(DataSamplingParallelFlow)
{
  WorkflowSpec workflow{
    { "producer",
      Inputs{},
      { OutputSpec{ "TPC", "CLUSTERS", 0 },
        OutputSpec{ "TPC", "CLUSTERS", 1 },
        OutputSpec{ "TPC", "CLUSTERS", 2 } },
      AlgorithmSpec{ [](ProcessingContext& ctx) {} } }
  };

  auto processingStages = parallel(
    DataProcessorSpec{
      "processingStage",
      Inputs{ { "dataTPC", "TPC", "CLUSTERS" } },
      Outputs{ { "TPC", "CLUSTERS_P" } },
      AlgorithmSpec{ [](ProcessingContext& ctx) {} } },
    3,
    [](DataProcessorSpec& spec, size_t index) {
      spec.inputs[0].subSpec = index;
      spec.outputs[0].subSpec = index;
    });

  workflow.insert(std::end(workflow), std::begin(processingStages), std::end(processingStages));

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSamplingDPL.json";
  DataSampling::GenerateInfrastructure(workflow, "json://" + configFilePath);

  for (int i = 0; i < 3; ++i) {
    auto disp = std::find_if(workflow.begin(), workflow.end(),
                             [i](const DataProcessorSpec& d) {
                               return d.name.find("Dispatcher") != std::string::npos;
                             });
    BOOST_REQUIRE(disp != workflow.end());

    auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                              [i](const InputSpec& in) {
                                return in.origin == DataOrigin("TPC") &&
                                       in.description == DataDescription("CLUSTERS") &&
                                       in.subSpec == i &&
                                       in.lifetime == Lifetime::Timeframe;
                              });
    BOOST_CHECK(input != disp->inputs.end());

    input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                         [i](const InputSpec& in) {
                           return in.origin == DataOrigin("TPC") &&
                                  in.description == DataDescription("CLUSTERS_P") &&
                                  in.subSpec == i &&
                                  in.lifetime == Lifetime::Timeframe;
                         });
    BOOST_CHECK(input != disp->inputs.end());

    auto output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                               [](const OutputSpec& out) {
                                 return out.origin == DataOrigin("DS") &&
                                        out.description == DataDescription("tpcclusters-0") &&
                                        out.subSpec == 0 &&
                                        out.lifetime == Lifetime::Timeframe;
                               });
    BOOST_CHECK(output != disp->outputs.end());

    output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                          [](const OutputSpec& out) {
                            return out.origin == DataOrigin("DS") &&
                                   out.description == DataDescription("tpcclusters-1") &&
                                   out.subSpec == 0 &&
                                   out.lifetime == Lifetime::Timeframe;
                          });
    BOOST_CHECK(output != disp->outputs.end());

    BOOST_CHECK(disp->algorithm.onInit != nullptr);
  }
}


BOOST_AUTO_TEST_CASE(DataSamplingTimePipelineFlow)
{
  WorkflowSpec workflow{
    { "producer",
      Inputs{},
      { OutputSpec{ "TPC", "CLUSTERS", 0, Lifetime::Timeframe } },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {} } },
    timePipeline(
      DataProcessorSpec{
        "processingStage",
        Inputs{
          { "dataTPC", "TPC", "CLUSTERS", 0, Lifetime::Timeframe } },
        Outputs{
          { "TPC", "CLUSTERS_P", 0, Lifetime::Timeframe } },
        AlgorithmSpec{
          [](ProcessingContext& ctx) {} } },
      3)
  };

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSamplingDPL.json";
  DataSampling::GenerateInfrastructure(workflow, "json://" + configFilePath, 3);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name.find("Dispatcher") != std::string::npos;
                           });

  BOOST_REQUIRE(disp != workflow.end());
  BOOST_CHECK_EQUAL(disp->inputs.size(), 2);
  BOOST_CHECK_EQUAL(disp->outputs.size(), 2);
  BOOST_CHECK(disp->algorithm.onInit != nullptr);
  BOOST_CHECK_EQUAL(disp->maxInputTimeslices, 3);
}


BOOST_AUTO_TEST_CASE(DataSamplingFairMq)
{
  WorkflowSpec workflow{
    specifyExternalFairMQDeviceProxy(
      "readout-proxy",
      Outputs{ { "TPC", "RAWDATA" } },
      "fake-channel-config",
      dataSamplingReadoutAdapter({ "TPC", "RAWDATA" }))
  };

  std::string configFilePath = std::string(getenv("O2_ROOT")) + "/share/tests/test_DataSamplingFairMQ.json";
  DataSampling::GenerateInfrastructure(workflow, "json://" + configFilePath);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name.find("Dispatcher") != std::string::npos;
                           });
  BOOST_REQUIRE(disp != workflow.end());

  auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                            [](const InputSpec& in) {
                              return in.origin == DataOrigin("TPC") &&
                                     in.description == DataDescription("RAWDATA") &&
                                     in.subSpec == 0 &&
                                     in.lifetime == Lifetime::Timeframe;
                            });
  BOOST_CHECK(input != disp->inputs.end());
  BOOST_CHECK_EQUAL(disp->outputs.size(), 1);

  auto channelConfig = std::find_if(disp->options.begin(), disp->options.end(),
                                    [](const ConfigParamSpec& opt) {
                                      return opt.name == "channel-config";
                                    });
  BOOST_REQUIRE(channelConfig != disp->options.end());
}


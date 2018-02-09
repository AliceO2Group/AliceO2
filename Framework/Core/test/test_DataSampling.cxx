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

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;
using DataOrigin = o2::header::DataOrigin;
using DataDescription = o2::header::DataDescription;

bool prepareConfigFile1(const std::string& path)
{
  std::ofstream cfgFile;
  cfgFile.open(path);
  if (!cfgFile.good()) {
    return false;
  }

  cfgFile <<
          "[DataSampling]\n"
          "tasksList=TpcQcTask\n"
          "enableTimePipeliningDispatchers=1\n"
          "enableParallelDispatchers=1\n"
          "enableProxy=0\n"
          "\n"
          "[TpcQcTask]\n"
          "taskDefinition=TpcQcTaskDefinition\n"
          "\n"
          "[TpcQcTaskDefinition]\n"
          "inputs=TpcClusters,TpcClustersProc\n"
          "fraction=0.1\n"
          "\n"
          "[TpcClusters]\n"
          "inputName=TPC_CLUSTERS_S\n"
          "dataOrigin=TPC\n"
          "dataDescription=CLUSTERS\n"
          "\n"
          "[TpcClustersProc]\n"
          "inputName=TPC_CLUSTERS_P_S\n"
          "dataOrigin=TPC\n"
          "dataDescription=CLUSTERS_P\n";

  cfgFile.close();
  return true;
}

bool prepareConfigFile2(const std::string& path)
{
  std::ofstream cfgFile;
  cfgFile.open(path);
  if (!cfgFile.good()) {
    return false;
  }

  cfgFile <<
          "[DataSampling]\n"
          "tasksList=FairQcTask\n"
          "enableTimePipeliningDispatchers=1\n"
          "enableParallelDispatchers=1\n"
          "enableProxy=0\n"
          "\n"
          "[FairQcTask]\n"
          "taskDefinition=FairQcTaskDefinition\n"
          "\n"
          "[FairQcTaskDefinition]\n"
          "inputs=fairTpcRaw\n"
          "fraction=0.2\n"
          "channelConfig=name=fairTpcRawOut,type=pub,method=bind,address=tcp://127.0.0.1:26525,rateLogging=1\n"
          "\n"
          "[fairTpcRaw]\n"
          "inputName=TPC_RAWDATA\n"
          "dataOrigin=TPC\n"
          "dataDescription=RAWDATA\n"
          "spawnConverter=1\n"
          "channelConfig=type=sub,method=connect,address=tcp://localhost:5558,rateLogging=1\n"
          "converterType=incrementalConverter";

  cfgFile.close();
  return true;
}

BOOST_AUTO_TEST_CASE(DataSamplingSimpleFlow)
{
  WorkflowSpec workflow{
    {
      "producer",
      Inputs{},
      {
        OutputSpec{"TPC", "CLUSTERS", 0, OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    },
    {
      "processingStage",
      Inputs{
        {"dataTPC", "TPC", "CLUSTERS", 0, InputSpec::Timeframe}
      },
      Outputs{
        {"TPC", "CLUSTERS_P", 0, OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    },
    {
      "qcTaskTpc",
      Inputs{
        {"TPC_CLUSTERS_S",   "TPC", "CLUSTERS_S",   0, InputSpec::Timeframe},
        {"TPC_CLUSTERS_P_S", "TPC", "CLUSTERS_P_S", 0, InputSpec::Timeframe}
      },
      Outputs{},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    }
  };

  std::string configFilePath = "/tmp/test_dataSamplingSimpleFlow.ini";
  BOOST_REQUIRE(prepareConfigFile1(configFilePath));
  DataSampling::GenerateInfrastructure(workflow, "file://" + configFilePath);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name == "Dispatcher0_for_TpcQcTask";
                           });
  BOOST_REQUIRE(disp != workflow.end());

  auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                            [](const InputSpec& in) {
                              return in.binding == "TPC_CLUSTERS_S" &&
                                     in.origin == DataOrigin("TPC") &&
                                     in.description == DataDescription("CLUSTERS") &&
                                     in.subSpec == 0 &&
                                     in.lifetime == InputSpec::Timeframe;
                            });
  BOOST_CHECK(input != disp->inputs.end());

  input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                       [](const InputSpec& in) {
                         return in.binding == "TPC_CLUSTERS_P_S" &&
                                in.origin == DataOrigin("TPC") &&
                                in.description == DataDescription("CLUSTERS_P") &&
                                in.subSpec == 0 &&
                                in.lifetime == InputSpec::Timeframe;
                       });
  BOOST_CHECK(input != disp->inputs.end());

  auto output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                             [](const OutputSpec& out) {
                               return out.origin == DataOrigin("TPC") &&
                                      out.description == DataDescription("CLUSTERS_P_S") &&
                                      out.subSpec == 0 &&
                                      out.lifetime == OutputSpec::Timeframe;
                             });
  BOOST_CHECK(output != disp->outputs.end());

  output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                        [](const OutputSpec& out) {
                          return out.origin == DataOrigin("TPC") &&
                                 out.description == DataDescription("CLUSTERS_S") &&
                                 out.subSpec == 0 &&
                                 out.lifetime == OutputSpec::Timeframe;
                        });
  BOOST_CHECK(output != disp->outputs.end());

  BOOST_CHECK(disp->algorithm.onProcess != nullptr);
}


BOOST_AUTO_TEST_CASE(DataSamplingParallelFlow)
{
  WorkflowSpec workflow{
    {
      "producer",
      Inputs{},
      {
        OutputSpec{"TPC", "CLUSTERS", 0, OutputSpec::Timeframe},
        OutputSpec{"TPC", "CLUSTERS", 1, OutputSpec::Timeframe},
        OutputSpec{"TPC", "CLUSTERS", 2, OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    },
    {
      "qcTaskTpc",
      Inputs{
        {"TPC_CLUSTERS_S",   "TPC", "CLUSTERS_S",   0, InputSpec::Timeframe},
        {"TPC_CLUSTERS_P_S", "TPC", "CLUSTERS_P_S", 0, InputSpec::Timeframe}
      },
      Outputs{},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    }
  };

  auto processingStages = parallel(
    DataProcessorSpec{
      "processingStage",
      Inputs{
        {"dataTPC", "TPC", "CLUSTERS", InputSpec::Timeframe}
      },
      Outputs{
        {"TPC", "CLUSTERS_P", OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    },
    3,
    [](DataProcessorSpec& spec, size_t index) {
      spec.inputs[0].subSpec = index;
      spec.outputs[0].subSpec = index;
    }
  );

  workflow.insert(std::end(workflow), std::begin(processingStages), std::end(processingStages));

  std::string configFilePath = "/tmp/test_dataSamplingParallel.ini";
  BOOST_REQUIRE(prepareConfigFile1(configFilePath));
  DataSampling::GenerateInfrastructure(workflow, "file://" + configFilePath);

  for (int i = 0; i < 3; ++i) {
    auto disp = std::find_if(workflow.begin(), workflow.end(),
                             [i](const DataProcessorSpec& d) {
                               return d.name == "Dispatcher" + std::to_string(i) + "_for_TpcQcTask";
                             });
    BOOST_REQUIRE(disp != workflow.end());

    auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                              [i](const InputSpec& in) {
                                return in.binding == "TPC_CLUSTERS_S" &&
                                       in.origin == DataOrigin("TPC") &&
                                       in.description == DataDescription("CLUSTERS") &&
                                       in.subSpec == i &&
                                       in.lifetime == InputSpec::Timeframe;
                              });
    BOOST_CHECK(input != disp->inputs.end());

    input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                         [i](const InputSpec& in) {
                           return in.binding == "TPC_CLUSTERS_P_S" &&
                                  in.origin == DataOrigin("TPC") &&
                                  in.description == DataDescription("CLUSTERS_P") &&
                                  in.subSpec == i &&
                                  in.lifetime == InputSpec::Timeframe;
                         });
    BOOST_CHECK(input != disp->inputs.end());

    auto output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                               [](const OutputSpec& out) {
                                 return out.origin == DataOrigin("TPC") &&
                                        out.description == DataDescription("CLUSTERS_P_S") &&
                                        out.subSpec == 0 &&
                                        out.lifetime == OutputSpec::Timeframe;
                               });
    BOOST_CHECK(output != disp->outputs.end());

    output = std::find_if(disp->outputs.begin(), disp->outputs.end(),
                          [](const OutputSpec& out) {
                            return out.origin == DataOrigin("TPC") &&
                                   out.description == DataDescription("CLUSTERS_S") &&
                                   out.subSpec == 0 &&
                                   out.lifetime == OutputSpec::Timeframe;
                          });
    BOOST_CHECK(output != disp->outputs.end());

    BOOST_CHECK(disp->algorithm.onProcess != nullptr);
  }
}


BOOST_AUTO_TEST_CASE(DataSamplingTimePipelineFlow)
{
  WorkflowSpec workflow{
    {
      "producer",
      Inputs{},
      {
        OutputSpec{"TPC", "CLUSTERS", 0, OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    },
    timePipeline(
      DataProcessorSpec{
        "processingStage",
        Inputs{
          {"dataTPC", "TPC", "CLUSTERS", 0, InputSpec::Timeframe}
        },
        Outputs{
          {"TPC", "CLUSTERS_P", 0, OutputSpec::Timeframe}
        },
        AlgorithmSpec{
          [](ProcessingContext& ctx) {}
        }
      }, 3),
    {
      "qcTaskTpc",
      Inputs{
        {"TPC_CLUSTERS_S",   "TPC", "CLUSTERS_S",   0, InputSpec::Timeframe},
        {"TPC_CLUSTERS_P_S", "TPC", "CLUSTERS_P_S", 0, InputSpec::Timeframe}
      },
      Outputs{},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {}
      }
    }
  };

  std::string configFilePath = "/tmp/test_dataSamplingTimePipeline.ini";
  BOOST_REQUIRE(prepareConfigFile1(configFilePath));
  DataSampling::GenerateInfrastructure(workflow, "file://" + configFilePath);

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name == "Dispatcher0_for_TpcQcTask";
                           });
  BOOST_REQUIRE(disp != workflow.end());
  BOOST_CHECK_EQUAL(disp->inputs.size(), 2);
  BOOST_CHECK_EQUAL(disp->outputs.size(), 2);
  BOOST_CHECK(disp->algorithm.onProcess != nullptr);
  BOOST_CHECK_EQUAL(disp->maxInputTimeslices, 3);
}


BOOST_AUTO_TEST_CASE(DataSamplingFairMq)
{
  WorkflowSpec workflow;

  std::string configFilePath = "/tmp/test_dataSamplingFairMq.ini";
  BOOST_REQUIRE(prepareConfigFile2(configFilePath));
  DataSampling::GenerateInfrastructure(workflow, "file://" + configFilePath);

  auto fairMqProxy = std::find_if(workflow.begin(), workflow.end(),
                                  [](const DataProcessorSpec& p) {
                                    return p.name == "FairMQ_proxy_for_FairQcTask";
                                  });
  BOOST_REQUIRE(fairMqProxy != workflow.end());

  auto output = std::find_if(fairMqProxy->outputs.begin(), fairMqProxy->outputs.end(),
                             [](const OutputSpec& out) {
                               return out.origin == DataOrigin("TPC") &&
                                      out.description == DataDescription("RAWDATA") &&
                                      out.subSpec == 0 &&
                                      out.lifetime == OutputSpec::Timeframe;
                             });
  BOOST_CHECK(output != fairMqProxy->outputs.end());

  auto disp = std::find_if(workflow.begin(), workflow.end(),
                           [](const DataProcessorSpec& d) {
                             return d.name == "Dispatcher0_for_FairQcTask";
                           });
  BOOST_REQUIRE(disp != workflow.end());

  auto input = std::find_if(disp->inputs.begin(), disp->inputs.end(),
                            [](const InputSpec& in) {
                              return in.binding == "TPC_RAWDATA" &&
                                     in.origin == DataOrigin("TPC") &&
                                     in.description == DataDescription("RAWDATA") &&
                                     in.subSpec == 0 &&
                                     in.lifetime == InputSpec::Timeframe;
                            });
  BOOST_CHECK(input != disp->inputs.end());
  BOOST_CHECK_EQUAL(disp->outputs.size(), 0);

  auto channelConfig = std::find_if(disp->options.begin(), disp->options.end(),
                                    [](const ConfigParamSpec& opt) {
                                      return opt.name == "channel-config";
                                    });
  BOOST_REQUIRE(channelConfig != disp->options.end());
}


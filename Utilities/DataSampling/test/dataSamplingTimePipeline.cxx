// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataSampling/DataSampling.h"

#include <thread>

using namespace o2::framework;
using namespace o2::utilities;

void customize(std::vector<CompletionPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}
void customize(std::vector<ChannelConfigurationPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

#include "Framework/InputSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "DataSampling/DataSampling.h"
#include "Framework/ParallelContext.h"
#include "Framework/runDataProcessing.h"

#include <boost/algorithm/string.hpp>

#include <chrono>
#include <iostream>

using namespace o2::framework;

struct FakeCluster {
  float x;
  float y;
  float z;
  float q;
};
using DataHeader = o2::header::DataHeader;

size_t parallelSize = 4;
size_t collectionChunkSize = 1000;
void someDataProducerAlgorithm(ProcessingContext& ctx);
void someProcessingStageAlgorithm(ProcessingContext& ctx);
void someSinkAlgorithm(ProcessingContext& ctx);

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  DataProcessorSpec dataProducer{
    "dataProducer",
    Inputs{},
    {
      OutputSpec{"TPC", "CLUSTERS"},
    },
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)someDataProducerAlgorithm}};

  auto processingStage = timePipeline(
    DataProcessorSpec{
      "processingStage",
      Inputs{
        {"dataTPC", "TPC", "CLUSTERS"}},
      Outputs{
        {"TPC", "CLUSTERS_P"}},
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback)someProcessingStageAlgorithm}},
    parallelSize);

  DataProcessorSpec sink{
    "sink",
    Inputs{
      {"dataTPC-proc", "TPC", "CLUSTERS_P", 0}},
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)someSinkAlgorithm}};

  // clang-format off
  DataProcessorSpec simpleQcTask{
    "simpleQcTask",
    Inputs{
      { "TPC_CLUSTERS_S",   { "DS", "simpleQcTask0" } },
      { "TPC_CLUSTERS_P_S", { "DS", "simpleQcTask1" } }
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [](ProcessingContext& ctx) {
        auto inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("TPC_CLUSTERS_S").payload);
        auto inputDataTpcProcessed = reinterpret_cast<const FakeCluster*>(ctx.inputs().get(
          "TPC_CLUSTERS_P_S").payload);

        const auto* header = o2::header::get<DataHeader*>(ctx.inputs().get("TPC_CLUSTERS_S").header);

        bool dataGood = true;
        for (int j = 0; j < header->payloadSize / sizeof(FakeCluster); ++j) {
          float diff = std::abs(-inputDataTpc[j].x - inputDataTpcProcessed[j].x) +
                       std::abs(2 * inputDataTpc[j].y - inputDataTpcProcessed[j].y) +
                       std::abs(inputDataTpc[j].z * inputDataTpc[j].q - inputDataTpcProcessed[j].z) +
                       std::abs(inputDataTpc[j].q - inputDataTpcProcessed[j].q);
          if (diff > 1) {
            dataGood = false;
            break;
          }
        }
        LOG(INFO) << "qcTaskTPC - received data is " << (dataGood ? "correct" : "wrong");
      }
    }
  };

  DataProcessorSpec dummyProducer{
    "dummy",
    Inputs{},
    Outputs{
      { {"tsthistos"}, "TST", "HISTOS", 0 },
      { {"tststring"}, "TST", "STRING", 0 }
    },
    AlgorithmSpec{[](ProcessingContext& ctx){}}
  };

  WorkflowSpec specs = {
    dataProducer,
    processingStage,
    sink,
    simpleQcTask,
    dummyProducer
  };
  const char* o2Root = getenv("O2_ROOT");
  if (o2Root == nullptr) {
    throw std::runtime_error("The O2_ROOT environment variable is not set, probably the O2 environment has not been loaded.");
  }
  std::string configurationSource = std::string("json:/") + o2Root + "/share/etc/exampleDataSamplingConfig.json";
  DataSampling::GenerateInfrastructure(specs, configurationSource);

  return specs;
}
// clang-format on

void someDataProducerAlgorithm(ProcessingContext& ctx)
{
  size_t index = ctx.services().get<ParallelContext>().index1D();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  // Creates a new message of size collectionChunkSize which
  // has "TPC" as data origin and "CLUSTERS" as data description.
  auto& tpcClusters = ctx.outputs().make<FakeCluster>(
    Output{"TPC", "CLUSTERS", static_cast<o2::header::DataHeader::SubSpecificationType>(index)}, collectionChunkSize);
  int i = 0;

  for (auto& cluster : tpcClusters) {
    assert(i < collectionChunkSize);
    cluster.x = index;
    cluster.y = i;
    cluster.z = i;
    cluster.q = rand() % 1000;
    i++;
  }
}

void someProcessingStageAlgorithm(ProcessingContext& ctx)
{
  size_t index = ctx.services().get<ParallelContext>().index1D();

  const FakeCluster* inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataTPC").payload);

  auto& processedTpcClusters = ctx.outputs().make<FakeCluster>(
    Output{"TPC", "CLUSTERS_P", static_cast<o2::header::DataHeader::SubSpecificationType>(index)},
    collectionChunkSize);

  int i = 0;
  for (auto& cluster : processedTpcClusters) {
    assert(i < collectionChunkSize);
    cluster.x = -inputDataTpc[i].x;
    cluster.y = 2 * inputDataTpc[i].y;
    cluster.z = inputDataTpc[i].z * inputDataTpc[i].q;
    cluster.q = inputDataTpc[i].q;
    i++;
  }
};

void someSinkAlgorithm(ProcessingContext& ctx)
{
  const FakeCluster* inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataTPC-proc").payload);
}

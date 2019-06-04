// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataSampling.h"

using namespace o2::framework;
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
#include "Framework/DataSampling.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ParallelContext.h"
#include "Framework/runDataProcessing.h"

#include <chrono>
#include <iostream>

#include <boost/algorithm/string.hpp>

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
  auto dataProducers = parallel(
    DataProcessorSpec{
      "dataProducer",
      Inputs{},
      { OutputSpec{ "TPC", "CLUSTERS" } },
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback)someDataProducerAlgorithm } },
    parallelSize,
    [](DataProcessorSpec& spec, size_t index) {
      DataSpecUtils::updateMatchingSubspec(spec.outputs[0], index);
    });

  auto processingStages = parallel(
    DataProcessorSpec{
      "processingStage",
      Inputs{
        { "dataTPC", "TPC", "CLUSTERS" } },
      Outputs{
        { "TPC", "CLUSTERS_P" } },
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback)someProcessingStageAlgorithm } },
    parallelSize,
    [](DataProcessorSpec& spec, size_t index) {
      DataSpecUtils::updateMatchingSubspec(spec.inputs[0], index);
      DataSpecUtils::updateMatchingSubspec(spec.outputs[0], index);
    });

  auto inputsSink = mergeInputs(
    { "dataTPC-proc", "TPC", "CLUSTERS_P" },
    parallelSize,
    [](InputSpec& input, size_t index) {
      DataSpecUtils::updateMatchingSubspec(input, index);
    });

  DataProcessorSpec sink{
    "sink",
    inputsSink,
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)someSinkAlgorithm }
  };

  // clang-format off
  DataProcessorSpec simpleQcTask{
    "simpleQcTask",
    Inputs{
      { "TPC_CLUSTERS_S",   "DS", "simpleQcTask-0", 0, Lifetime::Timeframe },
      { "TPC_CLUSTERS_P_S", "DS", "simpleQcTask-1", 0, Lifetime::Timeframe }
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [](ProcessingContext& ctx){
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

        LOG(INFO) << "simpleQcTask - received data is " << (dataGood ? "correct" : "wrong");
      }
    }
  };

  WorkflowSpec specs;
  specs.swap(dataProducers);
  specs.insert(std::end(specs), std::begin(processingStages), std::end(processingStages));
  specs.push_back(sink);
  specs.push_back(simpleQcTask);

  std::string configurationSource = std::string("json://") + getenv("BASEDIR") + "/../../O2/Framework/TestWorkflows/exampleDataSamplingConfig.json";
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
  auto tpcClusters = ctx.outputs().make<FakeCluster>(
    Output{ "TPC", "CLUSTERS", static_cast<o2::header::DataHeader::SubSpecificationType>(index) }, collectionChunkSize);
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
  auto processedTpcClusters = ctx.outputs().make<FakeCluster>(
    Output{ "TPC", "CLUSTERS_P", static_cast<o2::header::DataHeader::SubSpecificationType>(index) },
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

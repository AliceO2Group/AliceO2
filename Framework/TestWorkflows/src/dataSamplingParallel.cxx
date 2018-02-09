// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.


#include <iostream>
#include <boost/algorithm/string.hpp>

#include "Framework/InputSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSampling.h"
#include "Framework/ParallelContext.h"
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

struct FakeCluster {
  float x;
  float y;
  float z;
  float q;
};
using DataHeader = o2::Header::DataHeader;

size_t parallelSize = 4;
size_t collectionChunkSize = 1000;
void someDataProducerAlgorithm(ProcessingContext& ctx);
void someProcessingStageAlgorithm(ProcessingContext& ctx);
void someSinkAlgorithm(ProcessingContext& ctx);

void defineDataProcessing(std::vector<DataProcessorSpec>& specs)
{
  auto dataProducers = parallel(
    DataProcessorSpec{
      "dataProducer",
      Inputs{},
      {
        OutputSpec{"TPC", "CLUSTERS", OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback) someDataProducerAlgorithm
      }
    },
    parallelSize,
    [](DataProcessorSpec& spec, size_t index) {
      spec.outputs[0].subSpec = index;
    }
  );

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
        (AlgorithmSpec::ProcessCallback) someProcessingStageAlgorithm
      }
    },
    parallelSize,
    [](DataProcessorSpec& spec, size_t index) {
      spec.inputs[0].subSpec = index;
      spec.outputs[0].subSpec = index;
    }
  );


  auto inputsSink = mergeInputs(
    {"dataTPC-proc", "TPC", "CLUSTERS_P", InputSpec::Timeframe},
    parallelSize,
    [](InputSpec& input, size_t index) {
      input.subSpec = index;
    }
  );

  DataProcessorSpec sink{
    "sink",
    inputsSink,
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) someSinkAlgorithm
    }
  };


  DataProcessorSpec simpleQcTask{
    "simpleQcTask",
    Inputs{
      {"TPC_CLUSTERS_S",   "TPC", "CLUSTERS_S",   0, InputSpec::Timeframe},
      {"TPC_CLUSTERS_P_S", "TPC", "CLUSTERS_P_S", 0, InputSpec::Timeframe}
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [](ProcessingContext& ctx) {
        auto inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("TPC_CLUSTERS_S").payload);
        auto inputDataTpcProcessed = reinterpret_cast<const FakeCluster*>(ctx.inputs().get(
          "TPC_CLUSTERS_P_S").payload);

        const auto* header = o2::Header::get<DataHeader>(ctx.inputs().get("TPC_CLUSTERS_S").header);

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

  specs.swap(dataProducers);
  specs.insert(std::end(specs), std::begin(processingStages), std::end(processingStages));
  specs.push_back(sink);
  specs.push_back(simpleQcTask);

  std::string configurationSource = std::string("file://") + getenv("BASEDIR")
                                    + "/../../O2/Framework/TestWorkflows/exampleDataSamplerConfig.ini";

  DataSampling::GenerateInfrastructure(specs, configurationSource);
}


void someDataProducerAlgorithm(ProcessingContext& ctx)
{
  size_t index = ctx.services().get<ParallelContext>().index1D();
  sleep(1);
  // Creates a new message of size collectionChunkSize which
  // has "TPC" as data origin and "CLUSTERS" as data description.
  auto tpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS", index}, collectionChunkSize);
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
  auto processedTpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS_P", index},
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
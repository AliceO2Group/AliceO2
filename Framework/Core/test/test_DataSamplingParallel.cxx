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

#include <Configuration/ConfigurationInterface.h>
#include <Configuration/ConfigurationFactory.h>

#include <Framework/InputSpec.h>
#include <Framework/DataProcessorSpec.h>
#include <Framework/DataSampling.h>
#include <Framework/ParallelContext.h>
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
void someDataProducerAlgorithm(ProcessingContext &ctx);
void someProcessingStageAlgorithm (ProcessingContext &ctx);
void someSinkAlgorithm (ProcessingContext &ctx);

void defineDataProcessing(std::vector<DataProcessorSpec> &specs)
{

  auto dataProducers = parallel(
    DataProcessorSpec{
      "dataProducer",
      Inputs{},
      {
        OutputSpec{"TPC", "CLUSTERS", OutputSpec::Timeframe},
//        OutputSpec{"ITS", "CLUSTERS", OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback)someDataProducerAlgorithm
      }
    },
    parallelSize,
    [](DataProcessorSpec &spec, size_t index) {
      spec.outputs[0].subSpec = index;
//      spec.outputs[1].subSpec = index;
    }
  );

  auto processingStages = parallel(
    DataProcessorSpec{
      "processingStage",
      Inputs{
        {"dataTPC", "TPC", "CLUSTERS", InputSpec::Timeframe},
//        {"dataITS", "ITS", "CLUSTERS", InputSpec::Timeframe}
      },
      Outputs{
        {"TPC", "CLUSTERS_P", OutputSpec::Timeframe},
//        {"ITS", "CLUSTERS_P", OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        //CLion says it ambiguous without (AlgorithmSpec::ProcessCallback), but cmake compiles fine anyway.
        (AlgorithmSpec::ProcessCallback) someProcessingStageAlgorithm
      }
    },
    parallelSize,
    [](DataProcessorSpec &spec, size_t index) {
      spec.inputs[0].subSpec = index;
//      spec.inputs[1].subSpec = index;
      spec.outputs[0].subSpec = index;
//      spec.outputs[1].subSpec = index;
    }
  );


  auto inputsSink = mergeInputs(
    {"dataTPC-proc", "TPC", "CLUSTERS_P", InputSpec::Timeframe},
    parallelSize,
    [](InputSpec &input, size_t index) {
      input.subSpec = index;
    }
  );

//  auto itsInputsSink = mergeInputs(
//    {"dataITS-proc", "ITS", "CLUSTERS_P", InputSpec::Timeframe},
//    parallelSize,
//    [](InputSpec &input, size_t index) {
//      input.subSpec = index;
//    }
//  );
//
//  inputsSink.insert(std::end(inputsSink), std::begin(itsInputsSink), std::end(itsInputsSink));

  DataProcessorSpec sink{
    "sink",
    inputsSink,
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)someSinkAlgorithm
    }
  };


  DataProcessorSpec simpleQcTask{
    "simpleQcTask",
    Inputs{
      {"TPC_CLUSTERS_S", "TPC", "CLUSTERS_S", 0, InputSpec::Timeframe},
      {"TPC_CLUSTERS_P_S", "TPC", "CLUSTERS_P_S", 0, InputSpec::Timeframe}
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)[](ProcessingContext& ctx){
        auto inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("TPC_CLUSTERS_S").payload);
        auto inputDataTpcProcessed = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("TPC_CLUSTERS_P_S").payload);

        const auto *header = o2::Header::get<DataHeader>(ctx.inputs().get("TPC_CLUSTERS_S").header);

        bool dataGood = true;
        for (int j = 0; j < header->payloadSize/sizeof(FakeCluster) ; ++j) {
          float diff = std::abs(-inputDataTpc[j].x - inputDataTpcProcessed[j].x) +
                       std::abs(2*inputDataTpc[j].y - inputDataTpcProcessed[j].y) +
                       std::abs(inputDataTpc[j].z * inputDataTpc[j].q - inputDataTpcProcessed[j].z) +
                       std::abs(inputDataTpc[j].q - inputDataTpcProcessed[j].q);
          if ( diff > 1 ){
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

  //todo: get qcTasks list

  std::vector<std::string> taskNames = {"simpleQcTask"};
  //todo: get path as argument?
  std::string configurationSource = "file:///home/pkonopka/alice/O2/Framework/Core/test/exampleDataSamplerConfig.ini";

  DataSampling::GenerateInfrastructure(specs, configurationSource, taskNames);
}


void someDataProducerAlgorithm(ProcessingContext &ctx)
{
  size_t index = ctx.services().get<ParallelContext>().index1D();
  sleep(1);
  // Creates a new message of size collectionChunkSize which
  // has "TPC" as data origin and "CLUSTERS" as data description.
  auto tpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS", index}, collectionChunkSize);
  int i = 0;

  for (auto &cluster : tpcClusters) {
    assert(i < collectionChunkSize);
    cluster.x = index;
    cluster.y = i;
    cluster.z = i;
    cluster.q = rand() % 1000;
    i++;
  }

//  auto itsClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"ITS", "CLUSTERS", index}, collectionChunkSize);
//  i = 0;
//  for (auto &cluster : itsClusters) {
//    assert(i < collectionChunkSize);
//    cluster.x = index;
//    cluster.y = i;
//    cluster.z = i;
//    cluster.q = rand() % 10;
//    i++;
//  }
}


void someProcessingStageAlgorithm (ProcessingContext &ctx)
{
  size_t index = ctx.services().get<ParallelContext>().index1D();

  const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataTPC").payload);
//  const FakeCluster *inputDataIts = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataITS").payload);

  auto processedTpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS_P", index}, collectionChunkSize);
//  auto processedItsClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"ITS", "CLUSTERS_P", index}, collectionChunkSize);

  int i = 0;
  for(auto& cluster : processedTpcClusters){
    assert( i < collectionChunkSize);
    cluster.x = -inputDataTpc[i].x;
    cluster.y = 2*inputDataTpc[i].y;
    cluster.z = inputDataTpc[i].z * inputDataTpc[i].q;
    cluster.q = inputDataTpc[i].q;
    i++;
  }

//  i = 0;
//  for(auto& cluster : processedItsClusters){
//    assert( i < collectionChunkSize);
//    cluster.x = -inputDataIts[i].x;
//    cluster.y = 2*inputDataIts[i].y;
//    cluster.z = inputDataIts[i].z * inputDataIts[i].q;
//    cluster.q = inputDataIts[i].q;
//    i++;
//  }

};

void someSinkAlgorithm( ProcessingContext &ctx)
{
  const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataTPC-proc").payload);
//  const FakeCluster *inputDataIts = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataITS-proc").payload);
}
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include <random>
#include <Framework/ParallelContext.h>
#include "Framework/DataRefUtils.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include "Framework/MetricsService.h"
#include "FairMQLogger.h"

using namespace o2::framework;

struct FakeCluster {
    float x;
    float y;
    float z;
    float q;
};
using DataHeader = o2::Header::DataHeader;

size_t collectionChunkSize = 1000;
void someDataProducerAlgorithm(ProcessingContext &ctx);
void someProcessingStageAlgorithm (ProcessingContext &ctx);
void someSinkAlgorithm (ProcessingContext &ctx);

void defineDataProcessing(std::vector<DataProcessorSpec> &specs) {


  DataProcessorSpec dataProducer{
    "dataProducer",
    Inputs{},
    {
      OutputSpec{"TPC", "CLUSTERS", 0, OutputSpec::Timeframe},
      OutputSpec{"ITS", "CLUSTERS", 0, OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)someDataProducerAlgorithm
    }
  };

  DataProcessorSpec processingStage{
    "processingStage",
    Inputs{
      {"dataTPC", "TPC", "CLUSTERS", 0, InputSpec::Timeframe},
      {"dataITS", "ITS", "CLUSTERS", 0, InputSpec::Timeframe}
    },
    Outputs{
      {"TPC", "CLUSTERS_P", 0, OutputSpec::Timeframe},
      {"ITS", "CLUSTERS_P", 0, OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      //CLion says it ambiguous without (AlgorithmSpec::ProcessCallback), but cmake compiles fine anyway.
      (AlgorithmSpec::ProcessCallback) someProcessingStageAlgorithm
    }
  };


  DataProcessorSpec sink{
    "sink",
    Inputs{
      {"dataTPC-proc", "TPC", "CLUSTERS_P", 0, InputSpec::Timeframe},
      {"dataITS-proc", "ITS","CLUSTERS_P", 0, InputSpec::Timeframe}
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)someSinkAlgorithm
    }
  };

  DataProcessorSpec dataSampler{
    "dataSampler",
    Inputs{
      {"TPC_CLUSTERS", "TPC", "CLUSTERS", 0, InputSpec::Timeframe},
      {"ITS_CLUSTERS", "ITS", "CLUSTERS", 0, InputSpec::Timeframe},
      {"TPC_CLUSTERS_P", "TPC", "CLUSTERS_P", 0, InputSpec::Timeframe},
      {"ITS_CLUSTERS_P", "ITS", "CLUSTERS_P", 0, InputSpec::Timeframe}
    },
    Outputs{
      {"TPC", "CLUSTERS_S", 0, OutputSpec::Timeframe},
      {"ITS", "CLUSTERS_S", 0, OutputSpec::Timeframe},
      {"TPC", "CLUSTERS_P_S", 0, OutputSpec::Timeframe},
      {"ITS", "CLUSTERS_P_S", 0, OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      (AlgorithmSpec::InitCallback)[](InitContext &setup) {
        //It is an example dataSampler, that is not generic at all. It was supposed to be a test to see
        // what needs to be automatized.

        return (AlgorithmSpec::ProcessCallback)[](ProcessingContext &ctx) {

          const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("TPC_CLUSTERS").payload);
          const FakeCluster *inputDataIts = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("ITS_CLUSTERS").payload);
          const FakeCluster *inputProcessedDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("TPC_CLUSTERS_P").payload);
          const FakeCluster *inputProcessedDataIts = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("ITS_CLUSTERS_P").payload);

          //todo: whole %sampling needs to be done more efficient and generic

          unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
          std::default_random_engine generator(seed);
          std::bernoulli_distribution distribution(0.01);

          std::vector<FakeCluster> tpcToSend;
          std::vector<FakeCluster> processedTpcToSend;
          std::vector<FakeCluster> itsToSend;
          std::vector<FakeCluster> processedItsToSend;

          for (int i = 0; i < collectionChunkSize; ++i) {
            if (distribution(generator)) {
              tpcToSend.push_back(inputDataTpc[i]);
              processedTpcToSend.push_back(inputProcessedDataTpc[i]);
            }
            if (distribution(generator)) {
              itsToSend.push_back(inputDataIts[i]);
              processedItsToSend.push_back(inputProcessedDataIts[i]);
            }
          }

          //todo: find out what happens when size()==0
          auto sampledTpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS_S", 0}, tpcToSend.size());
          auto sampledProcessedTpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS_P_S", 0}, tpcToSend.size());
          auto sampledItsClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"ITS", "CLUSTERS_S", 0}, itsToSend.size());
          auto sampledProcessedItsClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"ITS", "CLUSTERS_P_S", 0}, itsToSend.size());

          //todo: more elegant way to copy
          int i = 0;
          for(auto& cluster : sampledTpcClusters){
            cluster = tpcToSend[i];
            i++;
          }
          i = 0;
          for(auto& cluster : sampledProcessedTpcClusters){
            cluster = processedTpcToSend[i];
            i++;
          }
          i = 0;
          for(auto& cluster : sampledItsClusters){
            cluster = itsToSend[i];
            i++;
          }
          i = 0;
          for(auto& cluster : sampledProcessedItsClusters){
            cluster = processedItsToSend[i];
            i++;
          }

//          LOG(INFO) << "dataSampler, TPC data size: " << tpcToSend.size();
        };
      }
    }
  };

  DataProcessorSpec qcTaskTpc{
    "qcTaskTpc",
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

//        LOG(INFO) << "qcTaskTPC, data payloadSize: " << header->payloadSize << " data count: " << header->payloadSize / sizeof(FakeCluster);

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

        LOG(INFO) << "qcTaskTPC - received data is " << (dataGood ? "correct" : "wrong");
      }
    }
  };

  DataProcessorSpec qcTaskIts{
    "qcTaskIts",
    Inputs{
      {"ITS_CLUSTERS_S", "ITS", "CLUSTERS_S", 0, InputSpec::Timeframe},
      {"ITS_CLUSTERS_P_S", "ITS", "CLUSTERS_P_S", 0, InputSpec::Timeframe}
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)[](ProcessingContext& ctx){

      }
    }
  };

  specs.push_back(dataProducer);
  specs.push_back(processingStage);
  specs.push_back(sink);

  specs.push_back(dataSampler);
  specs.push_back(qcTaskTpc);
  specs.push_back(qcTaskIts);
}

void someDataProducerAlgorithm(ProcessingContext &ctx)
{
  sleep(1);
  // Creates a new message of size collectionChunkSize which
  // has "TPC" as data origin and "CLUSTERS" as data description.
  auto tpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS", 0}, collectionChunkSize);
  int i = 0;

  for (auto &cluster : tpcClusters) {
    assert(i < collectionChunkSize);
    cluster.x = i;
    cluster.y = i;
    cluster.z = i;
    cluster.q = rand() % 1000;
    i++;
  }

  auto itsClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"ITS", "CLUSTERS", 0}, collectionChunkSize);
  i = 0;
  for (auto &cluster : itsClusters) {
    assert(i < collectionChunkSize);
    cluster.x = i;
    cluster.y = i;
    cluster.z = i;
    cluster.q = rand() % 10;
    i++;
  }
}


void someProcessingStageAlgorithm (ProcessingContext &ctx)
{
//  LOG(DEBUG) << "=========== void someProcessingStageAlgorithm (ProcessingContext &ctx) ===========";

  const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataTPC").payload);
  const FakeCluster *inputDataIts = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataITS").payload);


//  LOG(DEBUG) << "------------------processing stage input data------------------";
//  for (int i = 0; i < 5; i++){
//    LOG(DEBUG) << inputDataTpc[i].q << " " << inputDataTpc[i].x << " " << inputDataTpc[i].y << " " << inputDataTpc[i].z;
//  }

  auto processedTpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS_P", 0}, collectionChunkSize);
  auto processedItsClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"ITS", "CLUSTERS_P", 0}, collectionChunkSize);

  int i = 0;
  for(auto& cluster : processedTpcClusters){
    assert( i < collectionChunkSize);
    cluster.x = -inputDataTpc[i].x;
    cluster.y = 2*inputDataTpc[i].y;
    cluster.z = inputDataTpc[i].z * inputDataTpc[i].q;
    cluster.q = inputDataTpc[i].q;
    i++;
  }

  i = 0;
  for(auto& cluster : processedItsClusters){
    assert( i < collectionChunkSize);
    cluster.x = -inputDataIts[i].x;
    cluster.y = 2*inputDataIts[i].y;
    cluster.z = inputDataIts[i].z * inputDataIts[i].q;
    cluster.q = inputDataIts[i].q;
    i++;
  }

};

void someSinkAlgorithm( ProcessingContext &ctx)
{
  const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataTPC-proc").payload);
  const FakeCluster *inputDataIts = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataITS-proc").payload);

//  LOG(DEBUG) << "--------------------------sink input data---------------------------";
//  for (int i = 0; i < 5; i++){
//    LOG(DEBUG) << inputDataTpc[i].q << " " << inputDataTpc[i].x << " " << inputDataTpc[i].y << " " << inputDataTpc[i].z;
//  }
//
//  LOG(DEBUG) << "======================================";
}
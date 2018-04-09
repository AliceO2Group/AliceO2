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
#include "Framework/runDataProcessing.h"
#include "Framework/TMessageSerializer.h"
#include "FairMQLogger.h"
#include <TClonesArray.h>
#include <TH1F.h>
#include <TString.h>

using namespace o2::framework;

struct FakeCluster {
  float x;
  float y;
  float z;
  float q;
};
using DataHeader = o2::header::DataHeader;

size_t collectionChunkSize = 1000;
void someDataProducerAlgorithm(ProcessingContext& ctx);
void someProcessingStageAlgorithm(ProcessingContext& ctx);
void someSinkAlgorithm(ProcessingContext& ctx);

void defineDataProcessing(std::vector<DataProcessorSpec>& specs)
{
  DataProcessorSpec podDataProducer{
    "podDataProducer",
    Inputs{},
    {
      OutputSpec{ "TPC", "CLUSTERS", 0, OutputSpec::Timeframe },
      OutputSpec{ "ITS", "CLUSTERS", 0, OutputSpec::Timeframe }
    },
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) someDataProducerAlgorithm
    }
  };

  DataProcessorSpec processingStage{
    "processingStage",
    Inputs{
      { "dataTPC", "TPC", "CLUSTERS", 0, InputSpec::Timeframe },
      { "dataITS", "ITS", "CLUSTERS", 0, InputSpec::Timeframe }
    },
    Outputs{
      { "TPC", "CLUSTERS_P", 0, OutputSpec::Timeframe },
      { "ITS", "CLUSTERS_P", 0, OutputSpec::Timeframe }
    },
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) someProcessingStageAlgorithm
    }
  };


  DataProcessorSpec podSink{
    "podSink",
    Inputs{
      { "dataTPC-proc", "TPC", "CLUSTERS_P", 0, InputSpec::Timeframe },
      { "dataITS-proc", "ITS", "CLUSTERS_P", 0, InputSpec::Timeframe }
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) someSinkAlgorithm
    }
  };

  DataProcessorSpec qcTaskTpc{
    "qcTaskTpc",
    Inputs{
      { "TPC_CLUSTERS_S",   "TPC", "CLUSTERS_S",   0, InputSpec::Timeframe },
      { "TPC_CLUSTERS_P_S", "TPC", "CLUSTERS_P_S", 0, InputSpec::Timeframe }
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

  DataProcessorSpec rootDataProducer{
    "rootDataProducer",
    {},
    {
      OutputSpec{ "TST", "HISTOS", OutputSpec::Timeframe },
      OutputSpec{ "TST", "STRING", OutputSpec::Timeframe }
    },
    AlgorithmSpec{
      [](ProcessingContext& ctx) {
        sleep(1);
        // Create an histogram
        auto& singleHisto = ctx.outputs().make<TH1F>(OutputSpec{ "TST", "HISTOS", 0 },
                                                       "h1", "test", 100, -10., 10.);
        auto& aString = ctx.outputs().make<TObjString>(OutputSpec{ "TST", "STRING", 0 }, "foo");
        singleHisto.FillRandom("gaus", 1000);
        Double_t stats[4];
        singleHisto.GetStats(stats);
        LOG(INFO) << "sumw" << stats[0] << "\n"
                  << "sumw2" << stats[1] << "\n"
                  << "sumwx" << stats[2] << "\n"
                  << "sumwx2" << stats[3] << "\n";
      }
    }
  };

  DataProcessorSpec rootSink{
    "rootSink",
    {
      InputSpec{ "histos", "TST", "HISTOS", InputSpec::Timeframe },
      InputSpec{ "string", "TST", "STRING", InputSpec::Timeframe },
    },
    {},
    AlgorithmSpec{
      [](ProcessingContext& ctx) {
        auto h = ctx.inputs().get<TH1F>("histos");
        if (h.get() == nullptr) {
          throw std::runtime_error("Missing output");
        }
        Double_t stats[4];
        h->GetStats(stats);
        LOG(INFO) << "sumw" << stats[0] << "\n"
                  << "sumw2" << stats[1] << "\n"
                  << "sumwx" << stats[2] << "\n"
                  << "sumwx2" << stats[3] << "\n";
        auto s = ctx.inputs().get<TObjString>("string");

        LOG(INFO) << "String is " << s->GetString().Data();
      }
    }
  };

  DataProcessorSpec rootQcTask{
    "rootQcTask",
    {
      InputSpec{ "TST_HISTOS_S", "TST", "HISTOS_S", InputSpec::Timeframe },
      InputSpec{ "TST_STRING_S", "TST", "STRING_S", InputSpec::Timeframe },
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback) [](ProcessingContext& ctx) {
        auto h = ctx.inputs().get<TH1F>("TST_HISTOS_S");
        if (h.get() == nullptr) {
          throw std::runtime_error("Missing TST_HISTOS_S");
        }
        Double_t stats[4];
        h->GetStats(stats);
        LOG(INFO) << "sumw" << stats[0] << "\n"
                  << "sumw2" << stats[1] << "\n"
                  << "sumwx" << stats[2] << "\n"
                  << "sumwx2" << stats[3] << "\n";
        auto s = ctx.inputs().get<TObjString>("TST_STRING_S");

        LOG(INFO) << "qcTaskTst: TObjString is " << (std::string("foo") == s->GetString().Data() ? "correct" : "wrong");
      }
    }
  };

  specs.push_back(podDataProducer);
  specs.push_back(processingStage);
  specs.push_back(podSink);
  specs.push_back(qcTaskTpc);

  specs.push_back(rootDataProducer);
  specs.push_back(rootSink);
  specs.push_back(rootQcTask);

  std::string configurationSource = std::string("file://") + getenv("BASEDIR")
                                    + "/../../O2/Framework/TestWorkflows/exampleDataSamplerConfig.ini";

  DataSampling::GenerateInfrastructure(specs, configurationSource);
}


void someDataProducerAlgorithm(ProcessingContext& ctx)
{
  sleep(1);
  // Creates a new message of size collectionChunkSize which
  // has "TPC" as data origin and "CLUSTERS" as data description.
  auto tpcClusters = ctx.outputs().make<FakeCluster>(OutputSpec{ "TPC", "CLUSTERS", 0 }, collectionChunkSize);
  int i = 0;

  for (auto& cluster : tpcClusters) {
    assert(i < collectionChunkSize);
    cluster.x = i;
    cluster.y = i;
    cluster.z = i;
    cluster.q = rand() % 1000;
    i++;
  }

  auto itsClusters = ctx.outputs().make<FakeCluster>(OutputSpec{ "ITS", "CLUSTERS", 0 }, collectionChunkSize);
  i = 0;
  for (auto& cluster : itsClusters) {
    assert(i < collectionChunkSize);
    cluster.x = i;
    cluster.y = i;
    cluster.z = i;
    cluster.q = rand() % 10;
    i++;
  }
}


void someProcessingStageAlgorithm(ProcessingContext& ctx)
{
  const FakeCluster* inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataTPC").payload);
  const FakeCluster* inputDataIts = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataITS").payload);

  auto processedTpcClusters = ctx.outputs().make<FakeCluster>(OutputSpec{ "TPC", "CLUSTERS_P", 0 },
                                                                collectionChunkSize);
  auto processedItsClusters = ctx.outputs().make<FakeCluster>(OutputSpec{ "ITS", "CLUSTERS_P", 0 },
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

  i = 0;
  for (auto& cluster : processedItsClusters) {
    assert(i < collectionChunkSize);
    cluster.x = -inputDataIts[i].x;
    cluster.y = 2 * inputDataIts[i].y;
    cluster.z = inputDataIts[i].z * inputDataIts[i].q;
    cluster.q = inputDataIts[i].q;
    i++;
  }

};

void someSinkAlgorithm(ProcessingContext& ctx)
{
  const FakeCluster* inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataTPC-proc").payload);
  const FakeCluster* inputDataIts = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataITS-proc").payload);
}

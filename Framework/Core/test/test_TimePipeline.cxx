// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/InputSpec.h"
#include "Framework/DataProcessorSpec.h"
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

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {"dataProducer",
     Inputs{},
     {
       OutputSpec{"TPC", "CLUSTERS"},
     },
     AlgorithmSpec{
       (AlgorithmSpec::ProcessCallback)someDataProducerAlgorithm}},
    timePipeline(
      DataProcessorSpec{
        "processingStage",
        Inputs{
          {"dataTPC", "TPC", "CLUSTERS"}},
        Outputs{
          {"TPC", "CLUSTERS_P"}},
        AlgorithmSpec{
          (AlgorithmSpec::ProcessCallback)someProcessingStageAlgorithm}},
      parallelSize),
    DataProcessorSpec{
      "dataSampler",
      Inputs{
        {"dataTPC-sampled", "TPC", "CLUSTERS", 0, Lifetime::Timeframe},
      },
      Outputs{},
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback)[](ProcessingContext & ctx){}}}};
}

void someDataProducerAlgorithm(ProcessingContext& ctx)
{
  uint32_t index = ctx.services().get<ParallelContext>().index1D();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  // Creates a new message of size collectionChunkSize which
  // has "TPC" as data origin and "CLUSTERS" as data description.
  auto tpcClusters = ctx.outputs().make<FakeCluster>(Output{"TPC", "CLUSTERS", index}, collectionChunkSize);
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
  uint32_t index = ctx.services().get<ParallelContext>().index1D();

  const FakeCluster* inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataTPC").payload);

  auto processedTpcClusters =
    ctx.outputs().make<FakeCluster>(Output{"TPC", "CLUSTERS_P", index}, collectionChunkSize);

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

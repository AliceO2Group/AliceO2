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
#include "Framework/DataSpecUtils.h"
#include "Framework/ParallelContext.h"
#include "Framework/runDataProcessing.h"

#include <boost/algorithm/string.hpp>

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

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  std::vector<DataProcessorSpec> specs;
  auto dataProducers = parallel(
    DataProcessorSpec{
      "dataProducer",
      Inputs{},
      {OutputSpec{"TPC", "CLUSTERS", 0, Lifetime::Timeframe}},
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback)someDataProducerAlgorithm}},
    parallelSize,
    [](DataProcessorSpec& spec, size_t index) {
      DataSpecUtils::updateMatchingSubspec(spec.outputs[0], index);
    });

  auto processingStages = parallel(
    DataProcessorSpec{
      "processingStage",
      Inputs{
        {"dataTPC", "TPC", "CLUSTERS", 0, Lifetime::Timeframe}},
      Outputs{
        {"TPC", "CLUSTERS_P", 0, Lifetime::Timeframe}},
      AlgorithmSpec{
        //CLion says it ambiguous without (AlgorithmSpec::ProcessCallback), but cmake compiles fine anyway.
        (AlgorithmSpec::ProcessCallback)someProcessingStageAlgorithm}},
    parallelSize,
    [](DataProcessorSpec& spec, size_t index) {
      DataSpecUtils::updateMatchingSubspec(spec.inputs[0], index);
      DataSpecUtils::updateMatchingSubspec(spec.outputs[0], index);
    });

  auto inputsDataSampler = mergeInputs(
    {"dataTPC", "TPC", "CLUSTERS", 0, Lifetime::Timeframe},
    parallelSize,
    [](InputSpec& input, size_t index) {
      DataSpecUtils::updateMatchingSubspec(input, index);
    });
  auto inputsTpcProc = mergeInputs(
    {"dataTPC-proc", "TPC", "CLUSTERS_P", 0, Lifetime::Timeframe},
    parallelSize,
    [](InputSpec& input, size_t index) {
      DataSpecUtils::updateMatchingSubspec(input, index);
    });
  inputsDataSampler.insert(std::end(inputsDataSampler), std::begin(inputsTpcProc), std::end(inputsTpcProc));

  auto dataSampler = DataProcessorSpec{
    "dataSampler",
    inputsDataSampler,
    Outputs{
      {"TPC", "CLUSTERS_S"},
      {"TPC", "CLUSTERS_P_S"}},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)[](ProcessingContext & ctx){
        InputRecord& inputs = ctx.inputs();

  for (auto& input : inputs) {

    const InputSpec* inputSpec = input.spec;
    auto matcher = DataSpecUtils::asConcreteDataMatcher(*inputSpec);
    o2::header::DataDescription outputDescription = matcher.description;

    //todo: better sampled data flagging
    size_t len = strlen(outputDescription.str);
    if (len < outputDescription.size - 2) {
      outputDescription.str[len] = '_';
      outputDescription.str[len + 1] = 'S';
    }

    Output description{
      matcher.origin,
      outputDescription,
      0,
      inputSpec->lifetime};

    LOG(DEBUG) << "DataSampler sends data from subSpec: " << matcher.subSpec;

    const auto* inputHeader = o2::header::get<o2::header::DataHeader*>(input.header);
    auto output = ctx.outputs().make<char>(description, inputHeader->size());

    //todo: use some std function or adopt(), when it is available for POD data
    const char* input_ptr = input.payload;
    for (char& it : output) {
      it = *input_ptr++;
    }
  }
}
}
}
;

DataProcessorSpec qcTask{
  "qcTask",
  Inputs{
    {"dataTPC-sampled", "TPC", "CLUSTERS_S"},
    {"dataTPC-proc-sampled", "TPC", "CLUSTERS_P_S"}},
  Outputs{},
  AlgorithmSpec{
    (AlgorithmSpec::ProcessCallback)[](ProcessingContext & ctx){
      const FakeCluster* inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataTPC-sampled").payload);
const InputSpec* inputSpec = ctx.inputs().get("dataTPC-sampled").spec;
auto matcher = DataSpecUtils::asConcreteDataMatcher(*inputSpec);
LOG(DEBUG) << "qcTask received data with subSpec: " << matcher.subSpec;
}
}
}
;

DataProcessorSpec sink{
  "sink",
  mergeInputs(
    {"dataTPC-proc", "TPC", "CLUSTERS_P"},
    parallelSize,
    [](InputSpec& input, size_t index) {
      DataSpecUtils::updateMatchingSubspec(input, index);
    }),
  Outputs{},
  AlgorithmSpec{
    [](ProcessingContext& ctx) {
      const FakeCluster* inputDataTpc = reinterpret_cast<const FakeCluster*>(ctx.inputs().get("dataTPC-proc").payload);
    }}};

// error in qcTask:
specs.swap(dataProducers);
specs.insert(std::end(specs), std::begin(processingStages), std::end(processingStages));
specs.push_back(sink);
specs.push_back(dataSampler);
specs.push_back(qcTask);

// no error:
//  specs.swap(dataProducers);
//  specs.insert(std::end(specs), std::begin(processingStages), std::end(processingStages));
//  specs.push_back(dataSampler);
//  specs.push_back(qcTask);
//  specs.push_back(sink);

return specs;
}

void someDataProducerAlgorithm(ProcessingContext& ctx)
{
  size_t index = ctx.services().get<ParallelContext>().index1D();
  // Creates a new message of size collectionChunkSize which
  // has "TPC" as data origin and "CLUSTERS" as data description.
  auto tpcClusters = ctx.outputs().make<FakeCluster>(
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

  auto processedTpcClusters = ctx.outputs().make<FakeCluster>(
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

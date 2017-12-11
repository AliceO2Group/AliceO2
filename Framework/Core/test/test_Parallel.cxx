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

#include <Framework/InputSpec.h>
#include <Framework/DataProcessorSpec.h>
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

void defineDataProcessing(std::vector<DataProcessorSpec> &specs)
{

  auto dataProducers = parallel(
    DataProcessorSpec{
      "dataProducer",
      Inputs{},
      {
        OutputSpec{"TPC", "CLUSTERS", OutputSpec::Timeframe}
      },
      AlgorithmSpec{
        (AlgorithmSpec::ProcessCallback)someDataProducerAlgorithm
      }
    },
    parallelSize,
    [](DataProcessorSpec &spec, size_t index) {
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
        //CLion says it ambiguous without (AlgorithmSpec::ProcessCallback), but cmake compiles fine anyway.
        (AlgorithmSpec::ProcessCallback) someProcessingStageAlgorithm
      }
    },
    parallelSize,
    [](DataProcessorSpec &spec, size_t index) {
      spec.inputs[0].subSpec = index;
      spec.outputs[0].subSpec = index;
    }
  );

  auto inputsDataSampler = mergeInputs(
    {"dataTPC", "TPC", "CLUSTERS", InputSpec::Timeframe},
    parallelSize,
    [](InputSpec &input, size_t index) {
      input.subSpec = index;
    }
  );

  auto itsInputsSink = mergeInputs(
    {"dataITS-proc", "TPC", "CLUSTERS_P", InputSpec::Timeframe},
    parallelSize,
    [](InputSpec &input, size_t index) {
      input.subSpec = index;
    }
  );

  inputsDataSampler.insert(std::end(inputsDataSampler), std::begin(itsInputsSink), std::end(itsInputsSink));


  auto dataSampler = DataProcessorSpec{
    "dataSampler",
    inputsDataSampler,
    Outputs{
      {"TPC", "CLUSTERS_S", 0, OutputSpec::Timeframe},
      {"TPC", "CLUSTERS_P_S", 0, OutputSpec::Timeframe}
    },
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)[](ProcessingContext& ctx) {
        InputRecord& inputs = ctx.inputs();

        for(auto& input : inputs){

          const InputSpec* inputSpec = input.spec;
          o2::Header::DataDescription outputDescription = inputSpec->description;

          //todo: better sampled data flagging
          size_t len = strlen(outputDescription.str);
          if (len < outputDescription.size-2){
            outputDescription.str[len] = '_';
            outputDescription.str[len+1] = 'S';
          }

          OutputSpec outputSpec{inputSpec->origin,
                                outputDescription,
                                0,
                                static_cast<OutputSpec::Lifetime>(inputSpec->lifetime)};

          LOG(DEBUG) << "DataSampler sends data from subSpec: " << inputSpec->subSpec;

          const auto *inputHeader = o2::Header::get<o2::Header::DataHeader>(input.header);
          auto output = ctx.allocator().make<char>(outputSpec, inputHeader->size());

          //todo: use some std function or adopt(), when it is available for POD data
          const char* input_ptr = input.payload;
          for (char &it : output) {
            it = *input_ptr++;
          }
        }
      }
    }
  };

  DataProcessorSpec qcTask{
    "qcTask",
    Inputs{
      {"dataTPC-sampled", "TPC", "CLUSTERS_S", 0, InputSpec::Timeframe},
      {"dataTPC-proc-sampled", "TPC", "CLUSTERS_P_S", 0, InputSpec::Timeframe}
    },
    Outputs{},
    AlgorithmSpec{
      (AlgorithmSpec::ProcessCallback)[](ProcessingContext& ctx) {
        const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataTPC-sampled").payload);
        const InputSpec* inputSpec = ctx.inputs().get("dataTPC-sampled").spec;
        LOG(DEBUG) << "qcTask received data with subSpec: " << inputSpec->subSpec;
      }
    }
  };

  DataProcessorSpec sink{
    "sink",
    mergeInputs(
      {"dataTPC-proc", "TPC", "CLUSTERS_P", InputSpec::Timeframe},
      parallelSize,
      [](InputSpec &input, size_t index) {
        input.subSpec = index;
      }
    ),
    Outputs{},
    AlgorithmSpec{
      [](ProcessingContext &ctx){
        const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataTPC-proc").payload);
      }
    }
  };


  // error in qcTask:
  specs.swap(dataProducers);
  specs.insert(std::end(specs), std::begin(processingStages), std::end(processingStages));
  specs.push_back(sink);
  specs.push_back(dataSampler);
  specs.push_back(qcTask);


  //no error:
//  specs.swap(dataProducers);
//  specs.insert(std::end(specs), std::begin(processingStages), std::end(processingStages));
//  specs.push_back(dataSampler);
//  specs.push_back(qcTask);
//  specs.push_back(sink);

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
}


void someProcessingStageAlgorithm (ProcessingContext &ctx)
{
  size_t index = ctx.services().get<ParallelContext>().index1D();

  const FakeCluster *inputDataTpc = reinterpret_cast<const FakeCluster *>(ctx.inputs().get("dataTPC").payload);

  auto processedTpcClusters = ctx.allocator().make<FakeCluster>(OutputSpec{"TPC", "CLUSTERS_P", index}, collectionChunkSize);

  int i = 0;
  for(auto& cluster : processedTpcClusters){
    assert( i < collectionChunkSize);
    cluster.x = -inputDataTpc[i].x;
    cluster.y = 2*inputDataTpc[i].y;
    cluster.z = inputDataTpc[i].z * inputDataTpc[i].q;
    cluster.q = inputDataTpc[i].q;
    i++;
  }
};
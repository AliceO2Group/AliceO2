// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/InputSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/ParallelContext.h"
#include "Framework/ControlService.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ParallelContext.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DataRefUtils.h"
#include <fairmq/Device.h>
#include <algorithm>
#include <memory>
#include <unordered_map>

// customize clusterers and cluster decoders to process immediately what comes in
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // we customize the pipeline processors to consume data as it comes
  using CompletionPolicy = o2::framework::CompletionPolicy;
  using CompletionPolicyHelpers = o2::framework::CompletionPolicyHelpers;
  policies.push_back(CompletionPolicyHelpers::defineByName("consumer", CompletionPolicy::CompletionOp::Consume));
}
#include "Framework/runDataProcessing.h"

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

using DataHeader = o2::header::DataHeader;
using namespace o2::framework;

size_t nPipelines = 4;
size_t nParallelChannels = 6;
size_t nRolls = 1;

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  // define a template workflow with processors to be executed in a pipeline
  std::vector<DataProcessorSpec> workflowSpecs{
    {"processor1",
     Inputs{
       {"input", "TST", "TRIGGER", 0, Lifetime::Timeframe}},
     Outputs{
       {{"output"}, "TST", "PREPROC", 0, Lifetime::Timeframe}},
     AlgorithmSpec{[](ProcessingContext& ctx) {
       for (auto const& input : ctx.inputs()) {
         auto const& parallelContext = ctx.services().get<ParallelContext>();
         LOG(debug) << "instance " << parallelContext.index1D() << " of " << parallelContext.index1DSize() << ": "
                    << *input.spec << ": " << *((int*)input.payload);
         auto const* dataheader = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
         auto& data = ctx.outputs().make<int>(Output{"TST", "PREPROC", dataheader->subSpecification});
         ASSERT_ERROR(ctx.inputs().get<int>(input.spec->binding.c_str()) == parallelContext.index1D());
         data = parallelContext.index1D();
       }
     }}},
    {"processor2",
     Inputs{
       {"input", "TST", "PREPROC", 0, Lifetime::Timeframe}},
     Outputs{
       {{"output"}, "TST", "DATA", 0, Lifetime::Timeframe},
       {{"metadt"}, "TST", "META", 0, Lifetime::Timeframe}},
     AlgorithmSpec{[](ProcessingContext& ctx) {
       for (auto const& input : ctx.inputs()) {
         auto const& parallelContext = ctx.services().get<ParallelContext>();
         LOG(debug) << "instance " << parallelContext.index1D() << " of " << parallelContext.index1DSize() << ": "
                    << *input.spec << ": " << *((int*)input.payload);
         ASSERT_ERROR(ctx.inputs().get<int>(input.spec->binding.c_str()) == parallelContext.index1D());
         auto const* dataheader = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
         // TODO: there is a bug in the API for using OutputRef, returns an rvalue which can not be bound to
         // lvalue reference
         auto& data = ctx.outputs().make<int>(Output{"TST", "DATA", dataheader->subSpecification});
         data = ctx.inputs().get<int>(input.spec->binding.c_str());
         auto& meta = ctx.outputs().make<int>(Output{"TST", "META", dataheader->subSpecification});
         meta = dataheader->subSpecification;
       }
     }}},
  };

  // create parallel pipelines from the template workflow, the number of parallel channel is defined by
  // nParallelChannels and is distributed among the pipelines
  std::vector<o2::header::DataHeader::SubSpecificationType> subspecs(nParallelChannels);
  std::generate(subspecs.begin(), subspecs.end(), [counter = std::make_shared<int>(0)]() { return 0x1 << (*counter)++; });
  // correspondence between the subspec and the instance which serves this particular subspec
  // this is checked in the final consumer
  auto checkMap = std::make_shared<std::unordered_map<o2::header::DataHeader::SubSpecificationType, int>>();
  {
    size_t pipeline = 0;
    for (auto const& subspec : subspecs) {
      (*checkMap)[subspec] = pipeline;
      pipeline++;
      if (pipeline >= nPipelines) {
        pipeline = 0;
      }
    }
  }
  workflowSpecs = parallelPipeline(
    workflowSpecs, nPipelines,
    [&subspecs]() { return subspecs.size(); },
    [&subspecs](size_t index) { return subspecs[index]; });

  // define a producer process with outputs for all subspecs
  auto producerOutputs = [&subspecs]() {
    Outputs outputs;
    for (auto const& subspec : subspecs) {
      outputs.emplace_back("TST", "TRIGGER", subspec, Lifetime::Timeframe);
    }
    return outputs;
  };

  workflowSpecs.emplace_back(DataProcessorSpec{
    "trigger",
    Inputs{},
    producerOutputs(),
    AlgorithmSpec{[subspecs, counter = std::make_shared<int>(0)](ProcessingContext& ctx) {
      if (*counter < nRolls) {
        size_t pipeline = 0;
        size_t channels = subspecs.size();
        std::vector<size_t> multiplicities(nPipelines);
        for (pipeline = 0; pipeline < nPipelines; pipeline++) {
          multiplicities[pipeline] = channels / (nPipelines - pipeline) + ((channels % (nPipelines - pipeline)) > 0 ? 1 : 0);
          channels -= multiplicities[pipeline];
        }
        ASSERT_ERROR(channels == 0);
        size_t index = 0;
        auto end = subspecs.size();
        for (pipeline = 0; index < end; index++) {
          if (multiplicities[pipeline] == 0) {
            continue;
          }
          ctx.outputs().make<int>(Output{"TST", "TRIGGER", subspecs[index]}) = pipeline;
          multiplicities[pipeline++]--;
          if (pipeline >= nPipelines) {
            pipeline = 0;
          }
        }
        ASSERT_ERROR(index == subspecs.size());
        (*counter)++;
      }
      if (*counter == nRolls) {
        ctx.services().get<ControlService>().endOfStream();
        ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    }}});

  // the final consumer
  // map of bindings is used to check the channel names, note that the object is captured by
  // reference in mergeInputs which is a helper executed at construction of DataProcessorSpec,
  // while the AlgorithmSpec stores a lambda to be called later on, and the object must be
  // passed by copy or move in order to have a valid object upon invocation
  std::unordered_map<o2::header::DataHeader::SubSpecificationType, std::string> bindings;
  workflowSpecs.emplace_back(DataProcessorSpec{
    "consumer",
    mergeInputs({{"datain", "TST", "DATA", 0, Lifetime::Timeframe},
                 {"metain", "TST", "META", 0, Lifetime::Timeframe}},
                subspecs.size(),
                [&subspecs, &bindings](InputSpec& input, size_t index) {
                  input.binding += std::to_string(index);
                  DataSpecUtils::updateMatchingSubspec(input, subspecs[index]);
                  if (input.binding.compare(0, 6, "datain") == 0) {
                    bindings[subspecs[index]] = input.binding;
                  }
                }),
    Outputs(),
    AlgorithmSpec{adaptStateful([checkMap, bindings = std::move(bindings)](CallbackService& callbacks) {
      callbacks.set<CallbackService::Id::EndOfStream>([checkMap](EndOfStreamContext& ctx) {
        for (auto const& [subspec, pipeline] : *checkMap) {
          // we require all checks to be invalidated
          ASSERT_ERROR(pipeline == -1);
        }
        checkMap->clear();
      });
      callbacks.set<CallbackService::Id::Stop>([checkMap]() {
        ASSERT_ERROR(checkMap->size() == 0);
      });
      return adaptStateless([checkMap, bindings = std::move(bindings)](InputRecord& inputs) {
        bool haveDataIn = false;
        size_t index = 0;
        for (auto const& input : inputs) {
          if (!DataRefUtils::isValid(input)) {
            continue;
          }
          LOG(info) << "consuming : " << *input.spec << ": " << *((int*)input.payload);
          auto const* dataheader = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
          if (input.spec->binding.compare(0, 6, "datain") == 0) {
            if (input.spec->binding != bindings.at(dataheader->subSpecification)) {
              LOG(error) << "data with subspec " << dataheader->subSpecification << " at unexpected binding " << input.spec->binding << ", expected " << bindings.at(dataheader->subSpecification);
            }
            haveDataIn = true;
            ASSERT_ERROR(checkMap->at(dataheader->subSpecification) == inputs.get<int>(input.spec->binding.c_str()));
            // keep a backup before invalidating, the backup is used in the check below, which can throw and therefor
            // must be after invalidation
            auto pipeline = checkMap->at(dataheader->subSpecification);
            // invalidate, we check in the end of stream callback that all are invalidated
            (*checkMap)[dataheader->subSpecification] = -1;
            // check if we can access channels by binding
            if (inputs.isValid(bindings.at(dataheader->subSpecification))) {
              ASSERT_ERROR(inputs.get<int>(bindings.at(dataheader->subSpecification)) == pipeline);
            }
          }
        }
        // we require each input cycle to have data on datain channel
        ASSERT_ERROR(haveDataIn);
      });
    })}});

  return workflowSpecs;
}

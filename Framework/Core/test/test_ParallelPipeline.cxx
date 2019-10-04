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
#include "Framework/ControlService.h"
#include "Framework/ParallelContext.h"
#include <iostream>
#include <algorithm>
#include <memory>
#include <unordered_map>

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed)"; \
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
         LOG(DEBUG) << "instance " << parallelContext.index1D() << " of " << parallelContext.index1DSize() << ": "
                    << *input.spec << ": " << *((int*)input.payload);
         auto const* dataheader = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
         //auto data& = ctx.outputs().make<int>(OutputRef{"output", dataheader->subSpecification});
         auto& data = ctx.outputs().make<int>(Output{"TST", "PREPROC", dataheader->subSpecification, Lifetime::Timeframe});
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
         LOG(DEBUG) << "instance " << parallelContext.index1D() << " of " << parallelContext.index1DSize() << ": "
                    << *input.spec << ": " << *((int*)input.payload);
         ASSERT_ERROR(ctx.inputs().get<int>(input.spec->binding.c_str()) == parallelContext.index1D());
         auto const* dataheader = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
         // TODO: there is a bug in the API for using OutputRef, returns an rvalue which can not be bound to
         // lvalue reference
         //auto& data = ctx.outputs().make<int>(OutputRef{"output", dataheader->subSpecification});
         auto& data = ctx.outputs().make<int>(Output{"TST", "DATA", dataheader->subSpecification, Lifetime::Timeframe});
         data = ctx.inputs().get<int>(input.spec->binding.c_str());
         //auto meta& = ctx.outputs().make<int>(OutputRef{"metadt", dataheader->subSpecification});
         auto& meta = ctx.outputs().make<int>(Output{"TST", "META", dataheader->subSpecification, Lifetime::Timeframe});
         meta = dataheader->subSpecification;
       }
     }}},
  };

  // create parallel pipelines from the template workflow, the number of parallel channel is defined by
  // nParallelChannels and is distributed among the pipelines
  std::vector<o2::header::DataHeader::SubSpecificationType> subspecs(nParallelChannels);
  std::generate(subspecs.begin(), subspecs.end(), [counter = std::make_shared<int>(0)]() { return 0x1 << (*counter)++; });
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

  // we keep the correspondence between the subspec and the instance which serves this particular subspec
  // this is checked in the final consumer
  auto checkMap = std::make_shared<std::unordered_map<o2::header::DataHeader::SubSpecificationType, int>>();
  workflowSpecs.emplace_back(DataProcessorSpec{
    "trigger",
    Inputs{},
    producerOutputs(),
    AlgorithmSpec{[subspecs, checkMap, counter = std::make_shared<int>(0)](ProcessingContext& ctx) {
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
          ctx.outputs().make<int>(Output{"TST", "TRIGGER", subspecs[index], Lifetime::Timeframe}) = pipeline;
          (*checkMap)[subspecs[index]] = pipeline;
          multiplicities[pipeline++]--;
          if (pipeline >= nPipelines) {
            pipeline = 0;
          }
        }
        ASSERT_ERROR(index == subspecs.size());
        (*counter)++;
      }
      if (*counter == nRolls) {
        ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      }
    }}});

  // the final consumer
  workflowSpecs.emplace_back(DataProcessorSpec{
    "consumer",
    mergeInputs({{"datain", "TST", "DATA", 0, Lifetime::Timeframe},
                 {"metain", "TST", "META", 0, Lifetime::Timeframe}},
                subspecs.size(),
                [&subspecs](InputSpec& input, size_t index) {
                  DataSpecUtils::updateMatchingSubspec(input, subspecs[index]);
                }),
    Outputs(),
    AlgorithmSpec{[checkMap](ProcessingContext& ctx) {
      for (auto const& input : ctx.inputs()) {
        LOG(DEBUG) << "consuming : " << *input.spec << ": " << *((int*)input.payload);
        auto const* dataheader = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
        if (input.spec->binding.compare(0, 6, "datain") == 0) {
          ASSERT_ERROR((*checkMap)[dataheader->subSpecification] == ctx.inputs().get<int>(input.spec->binding.c_str()));
        }
      }
      ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
    }}});

  return workflowSpecs;
}

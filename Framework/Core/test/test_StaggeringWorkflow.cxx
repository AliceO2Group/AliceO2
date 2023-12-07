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

#include "Framework/WorkflowSpec.h"
#include "Framework/CallbackService.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataAllocator.h"
#include "Framework/RawDeviceService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ControlService.h"
#include "Framework/OutputRoute.h"
#include "Framework/Logger.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Output.h"
#include <cstring>
#include <regex>

void customize(std::vector<o2::framework::DispatchPolicy>& policies)
{
  // we customize all devices to dispatch data immediately
  auto producerMatcher = [](auto const& spec) {
    return std::regex_match(spec.name.begin(), spec.name.end(), std::regex("producer.*"));
  };
  auto processorMatcher = [](auto const& spec) {
    return std::regex_match(spec.name.begin(), spec.name.end(), std::regex("processor.*"));
  };
  auto triggerMatcher = [](auto const& query) {
    o2::framework::Output reference{"PROD", "TRIGGER"};
    return reference.origin == query.origin && reference.description == query.description;
  };
  policies.push_back({"producer-policy", producerMatcher, o2::framework::DispatchPolicy::DispatchOp::WhenReady, triggerMatcher});
  policies.push_back({"processor-policy", processorMatcher, o2::framework::DispatchPolicy::DispatchOp::WhenReady});
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // we customize the processors to consume data as it comes
  policies.push_back({"processor-consume",
                      [](o2::framework::DeviceSpec const& spec) {
                        // search for spec names starting with "processor"
                        return spec.name.find("processor") == 0;
                      },
                      [](auto const&) { return o2::framework::CompletionPolicy::CompletionOp::Consume; }});
}

#include "Framework/runDataProcessing.h"

using namespace o2::framework;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

constexpr size_t nPipelines = 3;
constexpr size_t nChannels = 10;

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  // first fill the subspecifications
  using MyDataType = o2::header::DataHeader::SubSpecificationType;
  std::vector<o2::header::DataHeader::SubSpecificationType> subspecs(nChannels);
  std::generate(subspecs.begin(), subspecs.end(), [counter = std::make_shared<int>(0)]() { return (*counter)++; });
  std::vector<OutputSpec> producerOutputs;
  for (auto const& subspec : subspecs) {
    producerOutputs.emplace_back(OutputSpec{"PROD", "CHANNEL", subspec, Lifetime::Timeframe});
    producerOutputs.emplace_back(OutputSpec{"PROD", "TRIGGER", subspec, Lifetime::Timeframe});
  }

  auto producerFct = adaptStateless([subspecs](DataAllocator& outputs, RawDeviceService& device, ControlService& control) {
    for (auto const& subspec : subspecs) {
      // since the snapshot copy is ready for sending it is scheduled but held back
      // because of the CompletionPolicy trigger matcher. This message will be
      // sent together with the second message.
      outputs.snapshot(Output{"PROD", "CHANNEL", subspec}, subspec);
      device.waitFor(100);
      outputs.snapshot(Output{"PROD", "TRIGGER", subspec}, subspec);
      device.waitFor(100);
    }
    control.endOfStream();
    control.readyToQuit(QuitRequest::Me);
  });

  auto processorFct = [](ProcessingContext& pc) {
    int nActiveInputs = 0;
    LOG(info) << "processing ...";
    for (auto const& input : pc.inputs()) {
      if (pc.inputs().isValid(input.spec->binding) == false) {
        // this input slot is empty
        continue;
      }
      auto& data = pc.inputs().get<MyDataType>(input.spec->binding.c_str());
      LOG(info) << "processing " << input.spec->binding << " " << data;
      // check if the channel binding starts with 'trigger'
      if (input.spec->binding.find("trigger") == 0) {
        pc.outputs().make<MyDataType>(Output{"PROC", "CHANNEL", data}) = data;
      }
      nActiveInputs++;
    }
    LOG(info) << "processed " << nActiveInputs << " inputs";
    // since we publish with delays, and two channels are always sent together
    ASSERT_ERROR(nActiveInputs == 2);
  };
  auto amendSinkInput = [subspecs](InputSpec& input, size_t index) {
    input.binding += std::to_string(subspecs[index]);
    DataSpecUtils::updateMatchingSubspec(input, subspecs[index]);
  };

  auto sinkFct = adaptStateful([](CallbackService& callbacks) {
    callbacks.set<CallbackService::Id::EndOfStream>([](EndOfStreamContext& context) {
      context.services().get<ControlService>().readyToQuit(QuitRequest::All);
    });
    return adaptStateless([](InputRecord& inputs) {
      for (auto const& input : inputs) {
        auto& data = inputs.get<MyDataType>(input.spec->binding.c_str());
        LOG(info) << "received channel " << data;
      } });
  });

  std::vector<DataProcessorSpec> workflow = parallelPipeline(
    std::vector<DataProcessorSpec>{DataProcessorSpec{
      "processor",
      {InputSpec{"input", "PROD", "CHANNEL", 0, Lifetime::Timeframe},
       InputSpec{"trigger", "PROD", "TRIGGER", 0, Lifetime::Timeframe}},
      {OutputSpec{"PROC", "CHANNEL", 0, Lifetime::Timeframe}},
      AlgorithmSpec{processorFct}}},
    nPipelines,
    [&subspecs]() { return subspecs.size(); },
    [&subspecs](size_t index) { return subspecs[index]; });

  workflow.emplace_back(DataProcessorSpec{
    "producer",
    Inputs{},
    producerOutputs,
    AlgorithmSpec{producerFct}});

  workflow.emplace_back(DataProcessorSpec{
    "sink",
    mergeInputs(InputSpec{"input", "PROC", "CHANNEL", 0, Lifetime::Timeframe}, nChannels, amendSinkInput),
    {},
    AlgorithmSpec{sinkFct}});
  return workflow;
}

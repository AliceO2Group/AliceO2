// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DataAllocator.h"
#include "Framework/InputRecord.h"
#include "Framework/InputSpec.h"
#include "Framework/OutputSpec.h"
#include "Framework/ControlService.h"
#include "Framework/OutputRoute.h"
#include "Framework/Logger.h"
#include "Framework/DispatchPolicy.h"
#include "Framework/DeviceSpec.h"
#include <chrono>
#include <cstring>
#include <iostream>

void customize(std::vector<o2::framework::DispatchPolicy>& policies)
{
  // we customize all devices to dispatch data immediately
  policies.push_back({"prompt-for-all", [](auto const&) { return true; }, o2::framework::DispatchPolicy::DispatchOp::WhenReady});
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
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed)"; \
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
  }

  auto producerFct = [subspecs](ProcessingContext& pc) {
    static bool ready = false;
    if (ready) {
      return;
    }
    for (auto const& subspec : subspecs) {
      //pc.outputs().make<MyDataType>(Output{ "PROD", "CHANNEL", subspec, Lifetime::Timeframe }) = subspec;
      pc.outputs().snapshot(Output{"PROD", "CHANNEL", subspec, Lifetime::Timeframe}, subspec);
      std::cout << "publishing channel " << subspec << std::endl;
      sleep(1);
    }
    ready = true;
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  };
  auto processorFct = [](ProcessingContext& pc) {
    int nActiveInputs = 0;
    for (auto const& input : pc.inputs()) {
      if (pc.inputs().isValid(input.spec->binding) == false) {
        // this input slot is empty
        continue;
      }
      auto& data = pc.inputs().get<MyDataType>(input.spec->binding.c_str());
      std::cout << "processing channel " << data << std::endl;
      pc.outputs().make<MyDataType>(Output{"PROC", "CHANNEL", data, Lifetime::Timeframe}) = data;
      nActiveInputs++;
    }
    // since we publish with delays, there should be only one active input at a time
    ASSERT_ERROR(nActiveInputs == 1);
  };
  auto amendSinkInput = [subspecs](InputSpec& input, size_t index) {
    input.binding += std::to_string(subspecs[index]);
    DataSpecUtils::updateMatchingSubspec(input, subspecs[index]);
  };
  auto sinkFct = [](ProcessingContext& pc) {
    for (auto const& input : pc.inputs()) {
      auto& data = pc.inputs().get<MyDataType>(input.spec->binding.c_str());
      std::cout << "received channel " << data << std::endl;
    }
    sleep(2);
    pc.services().get<ControlService>().readyToQuit(QuitRequest::All);
  };

  std::vector<DataProcessorSpec> workflow = parallelPipeline(
    std::vector<DataProcessorSpec>{DataProcessorSpec{
      "processor",
      {InputSpec{"input", "PROD", "CHANNEL", 0, Lifetime::Timeframe}},
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

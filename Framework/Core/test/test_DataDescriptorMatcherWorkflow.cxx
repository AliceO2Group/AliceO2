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

#include "Framework/DataDescriptorMatcher.h"
#include "Framework/DataDescriptorQueryBuilder.h"
#include "Framework/InputSpec.h"
#include "Framework/DataDescriptorMatcher.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/ControlService.h"
#include "Framework/InputRecord.h"
#include "Framework/DataAllocator.h"

// we need to add workflow options before including Framework/runDataProcessing
//void customize(std::vector<ConfigParamSpec>& workflowOptions)
//{
//}

#include "Framework/runDataProcessing.h"

// A test workflow for DataDescriptorNegator
// Create a processor which subscribes to input spec TST/SAMPLE/!0
// meaning TST/SAMPLE with all but subspec 0
// Subscribing processor to TST/SAMPLE/0

// Two tasks:
// - InputSpec matching to OutputSpec: OutputSpec has to options,
//   ConcreteDataMatcher and ConcreteDataTypeMatcher, DataSpecUtils has to
//   methods matching InputSpec matcher to either if this
//   -> this is sufficient for the use case if the negator is implemented
//      in DataDescriptorMatcher
// - matching of data packets to InputRoutes of the DataRelayer, also this
//   is based on DataDescriptorMatcher
//
// DataDescriptorMatcher extension
// - define Negator
//
// InputSpec definition:
// - refactor to have one constructor with templated parameters

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const& config)
{
  std::vector<DataProcessorSpec> workflow;

  auto producerCallback = [counter = std::make_shared<size_t>()](DataAllocator& outputs, ControlService& control) {
    if (*counter > 0) {
      // don't know if we enter the processing callback after the EOS was sent
      return;
    }
    outputs.make<unsigned int>(Output{"TST", "SAMPLE", 1}) = 1;
    outputs.make<unsigned int>(Output{"TST", "SAMPLE", 2}) = 2;
    ++(*counter);
    control.endOfStream();
  };

  workflow.emplace_back(DataProcessorSpec{"producer",
                                          {InputSpec{"timer", "TST", "TIMER", 0, Lifetime::Timer}},
                                          {OutputSpec{"TST", "SAMPLE", 1, Lifetime::Timeframe},
                                           OutputSpec{"TST", "SAMPLE", 2, Lifetime::Timeframe}},
                                          AlgorithmSpec{adaptStateless(producerCallback)},
                                          {ConfigParamSpec{"period-timer", VariantType::Int, 100000, {"timer"}}}});

  auto processorCallback = [counter = std::make_shared<size_t>()](InputRecord& inputs, DataAllocator& outputs) {
    // should not be called more than one time
    ASSERT_ERROR(*counter == 0);
    ++(*counter);
    int nBlocks = 0;
    for (auto ref : InputRecordWalker(inputs)) {
      auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
      ASSERT_ERROR(dh != nullptr);
      auto const& data = inputs.get<unsigned int>(ref);
      ASSERT_ERROR(data == dh->subSpecification);
      outputs.make<unsigned int>(OutputRef{"out", 0}) = data;
      LOG(info) << fmt::format("forwarded {}/{}/{} with data {}",
                               dh->dataOrigin.as<std::string>(),
                               dh->dataDescription.as<std::string>(),
                               dh->subSpecification,
                               data);
      ++nBlocks;
    }
    ASSERT_ERROR(nBlocks == 2);
  };

  using DataDescriptorMatcher = o2::framework::data_matcher::DataDescriptorMatcher;
  DataDescriptorMatcher processorInputMatcher = {
    DataDescriptorMatcher::Op::And,
    data_matcher::OriginValueMatcher{"TST"},
    std::make_unique<DataDescriptorMatcher>(
      DataDescriptorMatcher::Op::And,
      data_matcher::DescriptionValueMatcher{"SAMPLE"},
      std::make_unique<DataDescriptorMatcher>(
        DataDescriptorMatcher::Op::And,
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Not,
          data_matcher::SubSpecificationTypeValueMatcher{0}),
        std::make_unique<DataDescriptorMatcher>(
          DataDescriptorMatcher::Op::Just,
          data_matcher::StartTimeValueMatcher(data_matcher::ContextRef{0}))))};

  workflow.emplace_back(DataProcessorSpec{"processor",
                                          {InputSpec{"in", std::move(processorInputMatcher), Lifetime::Timeframe}},
                                          {OutputSpec{{"out"}, "TST", "SAMPLE", 0, Lifetime::Timeframe}},
                                          AlgorithmSpec{adaptStateless(processorCallback)}});

  auto sinkCallback = [counter = std::make_shared<size_t>()](InputRecord& inputs) {
    // should not be called more than one time
    ASSERT_ERROR(*counter == 0);
    ++(*counter);
    int nBlocks = 0;
    for (auto ref : InputRecordWalker(inputs)) {
      auto const* dh = DataRefUtils::getHeader<DataHeader*>(ref);
      ASSERT_ERROR(dh != nullptr);
      auto const& data = inputs.get<unsigned int>(ref);
      ASSERT_ERROR(data > 0 && data < 3);
      LOG(info) << fmt::format("received {}/{}/{} with data {}",
                               dh->dataOrigin.as<std::string>(),
                               dh->dataDescription.as<std::string>(),
                               dh->subSpecification,
                               data);
      ++nBlocks;
    }
    ASSERT_ERROR(nBlocks == 2);
  };

  workflow.emplace_back(DataProcessorSpec{"sink",
                                          {InputSpec{"in", "TST", "SAMPLE", 0, Lifetime::Timeframe}},
                                          {},
                                          AlgorithmSpec{adaptStateless(sinkCallback)}});

  return workflow;
}

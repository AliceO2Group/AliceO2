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

#ifndef ALICEO2_CUSTOMSTOPOLOGYCOMMON_H_
#define ALICEO2_CUSTOMSTOPOLOGYCOMMON_H_

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fairlogger/Logger.h>
#include <thread>

#include <Framework/CompletionPolicy.h>
#include <Framework/CompletionPolicyHelpers.h>
#include <Framework/ControlService.h>
#include <Mergers/CustomMergeableObject.h>
#include <Mergers/MergerBuilder.h>
#include <Mergers/MergerInfrastructureBuilder.h>
#include "common.h"

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  o2::mergers::MergerBuilder::customizeInfrastructure(policies);
  policies.emplace_back(o2::framework::CompletionPolicyHelpers::consumeWhenAny("data-checker"));
}

// keep this include here
#include <Framework/runDataProcessing.h>

namespace o2::framework
{
using SubSpecificationType = header::DataHeader::SubSpecificationType;

class CustomMergerTestGenerator
{
  static constexpr const char origin[] = {"TST"};
  static constexpr const char description[] = {"CUSTOM"};
  static constexpr const char description_moving_window[] = {"CUSTOM_MW"};

 public:
  CustomMergerTestGenerator(size_t expectedResult)
    : mExpectedResult{expectedResult}
  {
  }

  Inputs generateHistoProducers(WorkflowSpec& specs, size_t numberOfProducers)
  {
    Inputs inputs{};

    for (size_t p = 0; p < numberOfProducers; p++) {
      inputs.push_back({"mo", origin, description, static_cast<SubSpecificationType>(p + 1), Lifetime::Sporadic});

      DataProcessorSpec producer{
        "producer-custom" + std::to_string(p),
        Inputs{},
        Outputs{{{"mo"}, origin, description, static_cast<SubSpecificationType>(p + 1), Lifetime::Sporadic}},
        AlgorithmSpec{static_cast<AlgorithmSpec::ProcessCallback>([p, numberOfProducers](ProcessingContext& processingContext) mutable {
          auto customObject = std::make_unique<mergers::CustomMergeableObject>(1);
          auto subspec = static_cast<SubSpecificationType>(p + 1);
          processingContext.outputs().snapshot(OutputRef{"mo", subspec}, *customObject);
          processingContext.services().get<ControlService>().readyToQuit(QuitRequest::Me);
        })}};
      specs.push_back(producer);
    }

    return inputs;
  }

  void generateMergers(WorkflowSpec& specs, const Inputs& mergerInputs, mergers::InputObjectsTimespan mergerType)
  {
    using namespace mergers;

    MergerInfrastructureBuilder mergersBuilder;
    mergersBuilder.setInfrastructureName("custom");
    mergersBuilder.setInputSpecs(mergerInputs);
    mergersBuilder.setOutputSpec({{"main"}, origin, description, 0});

    MergerConfig config;
    config.inputObjectTimespan = {mergerType};
    std::vector<std::pair<size_t, size_t>> param = {{5, 1}};
    config.publicationDecision = {PublicationDecision::EachNSeconds, param};
    config.mergedObjectTimespan = {MergedObjectTimespan::FullHistory};
    config.topologySize = {TopologySize::NumberOfLayers, 2};

    if (mergerType != mergers::InputObjectsTimespan::FullHistory) {
      mergersBuilder.setOutputSpecMovingWindow({{"main"}, origin, description_moving_window, 0});
      config.publishMovingWindow = {PublishMovingWindow::Yes};
    }

    mergersBuilder.setConfig(config);

    mergersBuilder.generateInfrastructure(specs);
  }

  void generateCheckerIntegrating(WorkflowSpec& specs)
  {
    specs.push_back(DataProcessorSpec{
      "data-checker",
      Inputs{
        {"custom", origin, description, 0, Lifetime::Sporadic},
        {"custom_mw", origin, description_moving_window, 0, Lifetime::Sporadic},
      },
      Outputs{},
      AlgorithmSpec{
        AlgorithmSpec::InitCallback{[expectedResult = mExpectedResult](InitContext& initContext) {
          auto success = std::make_shared<bool>(false);
          mergers::test::registerCallbacksForTestFailure(initContext.services().get<CallbackService>(), success);

          return AlgorithmSpec::ProcessCallback{
            [expectedResult, numberOfCalls = 0, numberOfObjects = 0, numberOfMovingWindows = 0, lastObjectValue = 0, retries = 5, success](ProcessingContext& processingContext) mutable {
              numberOfCalls++;

              if (processingContext.inputs().isValid("custom")) {
                auto obj = processingContext.inputs().get<mergers::CustomMergeableObject*>("custom");
                numberOfObjects++;
                // we are keeping only the last value of secret, as there can be inconsitencies caused
                // by the lack of synchronisation between merger layers
                lastObjectValue = obj->getSecret();
              }

              if (processingContext.inputs().isValid("custom_mw")) {
                auto mw = processingContext.inputs().get<mergers::CustomMergeableObject*>("custom_mw");
                numberOfMovingWindows++;
                // it is not possible to check for correct value in moving window as the value can be lost due
                // to the sync between layers of mergers and movement of the window
              }

              if (numberOfCalls == retries) {
                processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);

                // we should get new object on each publish timeout of the mergers,
                // lower and upper boundaries of moving windows are chosen arbitrarily
                if (numberOfObjects != retries || numberOfMovingWindows == 0 || numberOfMovingWindows > 10) {
                  LOG(fatal) << "expected 5 objects and got: " << numberOfObjects << ", expected 1-10 moving windows and got: " << numberOfMovingWindows;
                  if (lastObjectValue != expectedResult) {
                    LOG(fatal) << "got wrong secret from object: " << lastObjectValue << ", expected: " << expectedResult;
                  }
                  return;
                }
                LOG(info) << "Received the expected objects, test successful";
                *success = true;
              }
            }};
        }}}});
  }

  void generateCheckerFullHistory(WorkflowSpec& specs)
  {
    specs.push_back(DataProcessorSpec{
      "data-checker",
      Inputs{
        {"custom", origin, description, 0, Lifetime::Sporadic},
      },
      Outputs{},
      AlgorithmSpec{
        AlgorithmSpec::InitCallback{[expectedResult = mExpectedResult](InitContext& initContext) {
          auto success = std::make_shared<bool>(false);
          mergers::test::registerCallbacksForTestFailure(initContext.services().get<CallbackService>(), success);

          return AlgorithmSpec::ProcessCallback{
            [expectedResult, retryNumber = 0, numberOfRetries = 5, success](ProcessingContext& processingContext) mutable {
              const auto obj = processingContext.inputs().get<mergers::CustomMergeableObject*>("custom");

              if (obj->getSecret() == expectedResult) {
                LOG(info) << "Received the expected object, test successful";
                *success = true;
                processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);
                return;
              }

              if (retryNumber++ == numberOfRetries) {
                processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);
                LOG(fatal) << "Unsuccessfully tried " << retryNumber << " times to get a expected result: " << expectedResult;
              }
            }};
        }}}});
  }

 private:
  size_t mExpectedResult;
};

} // namespace o2::framework

#endif

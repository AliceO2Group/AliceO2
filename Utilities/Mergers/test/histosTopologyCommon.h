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

#ifndef ALICEO2_HISTOSTOPOLOGYCOMMON_H_
#define ALICEO2_HISTOSTOPOLOGYCOMMON_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fairlogger/Logger.h>
#include <thread>

#include <TH1F.h>

#include <Framework/CompletionPolicy.h>
#include <Framework/CompletionPolicy.h>
#include <Framework/CompletionPolicy.h>
#include <Framework/CompletionPolicyHelpers.h>
#include <Framework/ControlService.h>
#include <Mergers/MergerInfrastructureBuilder.h>
#include <Mergers/MergerBuilder.h>

#include "common.h"

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  o2::mergers::MergerBuilder::customizeInfrastructure(policies);
}

// keep this include here
#include <Framework/runDataProcessing.h>

namespace o2::framework
{
using SubSpecificationType = header::DataHeader::SubSpecificationType;

template <size_t HistoSize>
class HistosMergerTestGenerator
{
  static constexpr const char origin[] = {"TST"};
  static constexpr const char description[] = {"HISTO"};

 public:
  HistosMergerTestGenerator(std::array<float, HistoSize>&& expectedResult, size_t histoBinsCount, double histoMin, double histoMax)
    : mExpectedResult{expectedResult}, mHistoBinsCount{histoBinsCount}, mHistoMin{histoMin}, mHistoMax{histoMax}
  {
  }

  Inputs generateHistoProducers(WorkflowSpec& specs, size_t numberOfProducers)
  {
    Inputs inputs{};
    for (size_t producerIdx = 1; producerIdx != numberOfProducers + 1; ++producerIdx) {
      inputs.push_back({"mo", origin, description, static_cast<SubSpecificationType>(producerIdx), Lifetime::Sporadic});
      specs.push_back(DataProcessorSpec{
        std::string{"producer-histo"} + std::to_string(producerIdx),
        Inputs{},
        Outputs{{{"mo"}, origin, description, static_cast<SubSpecificationType>(producerIdx), Lifetime::Sporadic}},
        AlgorithmSpec{
          static_cast<AlgorithmSpec::ProcessCallback>([histoBinsCount = mHistoBinsCount, histoMin = mHistoMin, histoMax = mHistoMax, producerIdx](ProcessingContext& processingContext) mutable {
            TH1F& histo = processingContext.outputs().make<TH1F>(
              Output{origin, description, static_cast<SubSpecificationType>(producerIdx)},
              "histo", "histo", histoBinsCount, histoMin, histoMax);
            histo.Fill(5);
            histo.Fill(producerIdx);
            processingContext.services().get<ControlService>().endOfStream();
            processingContext.services().get<ControlService>().readyToQuit(QuitRequest::Me);
          })}});
    }
    return inputs;
  }

  void generateMergers(WorkflowSpec& specs, const Inputs& producerInputs, mergers::InputObjectsTimespan mergerType)
  {
    using namespace mergers;

    MergerInfrastructureBuilder mergersBuilder;
    mergersBuilder.setInfrastructureName("histos");
    mergersBuilder.setInputSpecs(producerInputs);
    mergersBuilder.setOutputSpec({{"main"}, origin, description, 0});
    MergerConfig config;
    config.inputObjectTimespan = {mergerType};
    std::vector<std::pair<size_t, size_t>> param = {{5, 1}};
    config.publicationDecision = {PublicationDecision::EachNSeconds, param};
    config.mergedObjectTimespan = {MergedObjectTimespan::FullHistory};
    config.topologySize = {TopologySize::NumberOfLayers, 2};
    mergersBuilder.setConfig(config);

    mergersBuilder.generateInfrastructure(specs);
  }

  void generateChecker(WorkflowSpec& specs)
  {
    specs.push_back(DataProcessorSpec{
      "data-checker",
      Inputs{{"histo", origin, description, 0, Lifetime::Sporadic}},
      Outputs{},
      AlgorithmSpec{
        AlgorithmSpec::InitCallback{[expectedResult = mExpectedResult](InitContext& initContext) {
          auto success = std::make_shared<bool>(false);
          mergers::test::registerCallbacksForTestFailure(initContext.services().get<CallbackService>(), success);

          // reason for this crude retry is that multiple layers are not synchronized between each other and publish on their own timers,
          // number of retries is chosen arbitrarily as we need to retry at least twice
          return AlgorithmSpec::ProcessCallback{[expectedResult, retryNumber = 1, retries = 5, success](ProcessingContext& processingContext) mutable {
            const auto histo = processingContext.inputs().get<TH1F*>("histo");

            LOG(info) << "RETRY: " << retryNumber << ": comparing: " << std::to_string(histo) << " to the expected: " << std::to_string(expectedResult);
            if (std::equal(expectedResult.begin(), expectedResult.end(), histo->GetArray(), histo->GetArray() + histo->GetSize())) {
              LOG(info) << "Received the expected object, test successful";
              *success = true;
              processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);
              return;
            }

            if (retryNumber++ >= retries) {
              processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);
              LOG(fatal) << "received incorrect data: " << std::to_string(histo) << ", expected: " << std::to_string(gsl::span(expectedResult));
            }
          }};
        }}}});
  }

 private:
  std::array<float, HistoSize> mExpectedResult;
  size_t mHistoBinsCount;
  double mHistoMin;
  double mHistoMax;
};

} // namespace o2::framework

#endif

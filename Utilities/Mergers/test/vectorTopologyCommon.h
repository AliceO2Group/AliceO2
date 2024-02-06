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

#ifndef ALICEO2_VECTORTOPOLOGYCOMMON_H_
#define ALICEO2_VECTORTOPOLOGYCOMMON_H_

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
#include <Mergers/ObjectStore.h>

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
class VectorMergerTestGenerator
{
  static constexpr const char origin[] = {"TST"};
  static constexpr const char description[] = {"VEC"};

 public:
  VectorMergerTestGenerator(std::vector<std::array<float, HistoSize>>&& expectedResult, size_t histoBinsCount, double histoMin, double histoMax)
    : mExpectedResult{expectedResult}, mHistoBinsCount{histoBinsCount}, mHistoMin{histoMin}, mHistoMax{histoMax}
  {
  }

  Inputs generateHistoProducers(WorkflowSpec& specs, size_t numberOfProducers)
  {
    Inputs mergersInputs;

    for (size_t producerIdx = 1; producerIdx < numberOfProducers + 1; ++producerIdx) {
      mergersInputs.push_back({"mo", origin, description, static_cast<SubSpecificationType>(producerIdx), Lifetime::Sporadic});
      DataProcessorSpec producer{
        "producer-vec" + std::to_string(producerIdx),
        Inputs{},
        Outputs{{{"mo"}, origin, description, static_cast<SubSpecificationType>(producerIdx), Lifetime::Sporadic}},
        AlgorithmSpec{static_cast<AlgorithmSpec::ProcessCallback>([producerIdx, numberOfProducers, binsCount = mHistoBinsCount, histoMin = mHistoMin, histoMax = mHistoMax, sent = false](ProcessingContext& processingContext) mutable {
          if (sent) {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
            return;
          }

          const auto subspec = static_cast<SubSpecificationType>(producerIdx);
          auto vectorOfHistos = std::make_unique<mergers::VectorOfRawTObjects>(2);

          int i = 0;
          for (auto& hist_ptr : *vectorOfHistos) {
            const auto histoname = std::string{"histo"} + std::to_string(++i);
            auto* hist = new TH1F(histoname.c_str(), histoname.c_str(), binsCount, histoMin, histoMax);
            hist->Fill(producerIdx);
            hist->Fill(5);
            hist_ptr = hist;
          }

          processingContext.outputs().snapshot(OutputRef{"mo", subspec}, *vectorOfHistos);
          for_each(vectorOfHistos->begin(), vectorOfHistos->end(), [](auto& histoPtr) { delete histoPtr; });
          sent = true;
        })}};
      specs.push_back(producer);
    }
    return mergersInputs;
  }

  void generateMergers(WorkflowSpec& specs, const Inputs& mergersInputs, mergers::InputObjectsTimespan mergerType)
  {
    using namespace mergers;

    MergerInfrastructureBuilder mergersBuilder;
    mergersBuilder.setInfrastructureName("vec");
    mergersBuilder.setInputSpecs(mergersInputs);
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
      Inputs{{"vec", origin, description, 0, Lifetime::Sporadic}},
      Outputs{},
      AlgorithmSpec{
        AlgorithmSpec::InitCallback{[expectedResult = mExpectedResult](InitContext&) {
          // reason for this crude retry is that multiple layers are not synchronized between each other and publish on their own timers
          return AlgorithmSpec::ProcessCallback{[expectedResult, retryNumber = 0, retries = 5](ProcessingContext& processingContext) mutable {
            if (retryNumber++ == retries) {
              processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);
              LOG(fatal) << "received wrong data\n";
              return;
            }

            auto dataRef = processingContext.inputs().get("vec");
            auto vectorOfHistos = DataRefUtils::as<ROOTSerialized<std::vector<TObject*>>>(dataRef);

            if (vectorOfHistos->size() == expectedResult.size()) {
              size_t resultIdx = 0;
              for (const auto histo : *vectorOfHistos) {
                if (!VectorMergerTestGenerator<HistoSize>::compareHistoToExpected(expectedResult[resultIdx++], *dynamic_cast<TH1F*>(histo))) {
                  return;
                }
              }
            } else {
              return;
            }

            processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);
          }};
        }}}});
  }

 private:
  static bool compareHistoToExpected(const std::array<float, HistoSize>& expected, const TH1F& histo)
  {
    return gsl::span{expected} == gsl::span(histo.GetArray(), histo.GetSize());
  }

  std::vector<std::array<float, HistoSize>> mExpectedResult;
  size_t mHistoBinsCount;
  double mHistoMin;
  double mHistoMax;
};

} // namespace o2::framework

#endif

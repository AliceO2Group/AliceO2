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

/// \file A unit test of mergers.cxx
/// \brief
///
/// \author Michal Tichak, michal.tichak@cern.ch

#include <chrono>
#include <cstdlib>
#include <sstream>
#include <thread>
#include "Mergers/MergerBuilder.h"
#include <Framework/CompletionPolicy.h>
#include <Framework/CompletionPolicyHelpers.h>
#include <TH1F.h>
#include <fairlogger/Logger.h>
#include "Framework/ControlService.h"

using namespace o2::framework;
using namespace o2::mergers;

void customize(std::vector<CompletionPolicy>& policies)
{
  MergerBuilder::customizeInfrastructure(policies);
}

#include "Framework/runDataProcessing.h"
#include "Mergers/MergerInfrastructureBuilder.h"
#include <Framework/CompletionPolicy.h>

using SubSpecificationType = o2::header::DataHeader::SubSpecificationType;
constexpr size_t producersAmount = 2;

void print_histo(const TH1F& h)
{
  std::stringstream ss;
  for (size_t i = 0; i != h.GetSize(); ++i) {
    ss << h[i] << " ";
  }
  LOG(info) << ss.str();
}

template <typename T>
void print_histo(const T& histo_ptr)
{
  print_histo(*histo_ptr.get());
}

constexpr size_t binsCount = 10;
constexpr double min = 0;
constexpr double max = 10;

constexpr std::array<float, 12> expectedVector{
  1.,
  1.,
  1.,
  0.,
  0.,
  0.,
  2.,
  0.,
  0.,
  0.,
  0.,
  0.,
};

bool compareHistoToExpected(const TH1F& histo)
{
  return gsl::span{expectedVector} == gsl::span(histo.GetArray(), histo.GetSize());
}

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec specs;

  Inputs mergersInputs;
  for (size_t p = 0; p < producersAmount; p++) {
    mergersInputs.push_back({"mo", "TST", "HISTO", static_cast<SubSpecificationType>(p + 1), Lifetime::Sporadic});
    DataProcessorSpec producer{
      "producer-histo" + std::to_string(p),
      Inputs{},
      Outputs{{{"mo"}, "TST", "HISTO", static_cast<SubSpecificationType>(p + 1), Lifetime::Sporadic}},
      AlgorithmSpec{static_cast<AlgorithmSpec::ProcessCallback>([p, dataSent = false](ProcessingContext& processingContext) mutable {
        if (dataSent) {
          std::this_thread::sleep_for(std::chrono::milliseconds{100});
          return;
        }
        auto subspec = static_cast<SubSpecificationType>(p + 1);
        TH1F& histo = processingContext.outputs().make<TH1F>(Output{"TST", "HISTO", subspec}, "histo", "histo", binsCount, min, max);
        histo.Fill(5);
        histo.Fill(p);
        print_histo(histo);
        dataSent = true;
      })}};
    specs.push_back(producer);
  }

  MergerInfrastructureBuilder mergersBuilder;
  mergersBuilder.setInfrastructureName("histos");
  mergersBuilder.setInputSpecs(mergersInputs);
  mergersBuilder.setOutputSpec({{"main"}, "TST", "HISTO", 0});
  MergerConfig config;
  config.inputObjectTimespan = {InputObjectsTimespan::FullHistory};
  std::vector<std::pair<size_t, size_t>> param = {{5, 1}};
  config.publicationDecision = {PublicationDecision::EachNSeconds, param};
  config.mergedObjectTimespan = {MergedObjectTimespan::FullHistory};
  config.topologySize = {TopologySize::NumberOfLayers, 2};
  mergersBuilder.setConfig(config);

  mergersBuilder.generateInfrastructure(specs);

  DataProcessorSpec printer{
    "data-checker",
    Inputs{{"histo", "TST", "HISTO", 0, Lifetime::Sporadic}},
    Outputs{},
    AlgorithmSpec{
      AlgorithmSpec::InitCallback{[](InitContext&) {
        return AlgorithmSpec::ProcessCallback{[](ProcessingContext& processingContext) mutable {
          auto histo = processingContext.inputs().get<TH1F*>("histo");
          print_histo(*histo);
          processingContext.services().get<ControlService>().readyToQuit(QuitRequest::All);
          if (!compareHistoToExpected(*histo.get())) {
            LOG(fatal) << "received incorrect data";
          }
        }};
      }}}};
  specs.push_back(printer);
  return specs;
}

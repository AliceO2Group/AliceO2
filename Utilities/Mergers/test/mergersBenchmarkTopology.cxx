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

///
/// \file    mergersBenchmarkTopology.cxx
/// \author  Piotr Konopka
///
/// \brief This is a DPL workflow to benchmark Mergers

#include "Framework/RootSerializationSupport.h"
#include "Mergers/MergerBuilder.h"

#include <Framework/CompletionPolicy.h>

using namespace o2::framework;
using namespace o2::mergers;

void customize(std::vector<CompletionPolicy>& policies)
{
  MergerBuilder::customizeInfrastructure(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& options)
{
  options.push_back({"obj-bins", VariantType::Int, 100, {"Number of bins in a histogram"}});
  options.push_back({"obj-rate", VariantType::Double, 1.0, {"Number of objects per second sent by one producer"}});
  options.push_back({"obj-producers", VariantType::Int, 4, {"Number of objects producers"}});

  options.push_back({"mergers-layers", VariantType::Int, 2, {"Number of layers in the merger topology"}});
  options.push_back({"mergers-publication-interval", VariantType::Double, 10.0, {"Publication interval of merged object [s]. It takes effect with --mergers-publication-decision interval"}});
  options.push_back(
    {"mergers-input-timespan", VariantType::String, "diffs", {"Should the topology use 'diffs' or 'full' objects"}});
}

#include "Framework/runDataProcessing.h"

#include <TH1F.h>
#include <memory>
#include <random>
#include "Framework/Logger.h"
#include "Mergers/MergerInfrastructureBuilder.h"

using namespace std::chrono;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  int objectsBins = config.options().get<int>("obj-bins");
  double objectsRate = config.options().get<double>("obj-rate");
  int objectsProducers = config.options().get<int>("obj-producers");

  int mergersLayers = config.options().get<int>("mergers-layers");
  PublicationDecision mergersPublicationDecision = PublicationDecision::EachNSeconds;
  double mergersPublicationInterval = config.options().get<double>("mergers-publication-interval");
  InputObjectsTimespan mergersInputObjectTimespan =
    config.options().get<std::string>("mergers-input-timespan") == "full" ? InputObjectsTimespan::FullHistory : InputObjectsTimespan::LastDifference;

  WorkflowSpec specs;
  // clang-format off
  // one 1D histo, binwise
  {
    Inputs mergersInputs;
    for (size_t p = 0; p < objectsProducers; p++) {
      mergersInputs.push_back({ "mo",               "TST",
                                "HISTO",            static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1),
                                Lifetime::Timeframe });
      DataProcessorSpec producer{
        "producer-histo" + std::to_string(p), Inputs{},
        Outputs{ { { "mo" },
                   "TST",
                   "HISTO",
                   static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1),
                   Lifetime::Timeframe } },
        AlgorithmSpec{
          (AlgorithmSpec::ProcessCallback)[ p, periodus = int(1000000 / objectsRate), objectsBins, objectsProducers ](
            ProcessingContext& processingContext) mutable { static auto lastTime = steady_clock::now();
            auto now = steady_clock::now();

            if (duration_cast<microseconds>(now - lastTime).count() > periodus) {

              lastTime += microseconds(periodus);

              auto subspec = static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1);
              TH1F& histo = processingContext.outputs().make<TH1F>(Output{ "TST", "HISTO", subspec }, "gauss", "gauss", objectsBins, -3, 3);
              histo.FillRandom("gaus", 10000);
            }
          }
        }
      };
      specs.push_back(producer);
    }

    MergerInfrastructureBuilder mergersBuilder;
    mergersBuilder.setInfrastructureName("histos");
    mergersBuilder.setInputSpecs(mergersInputs);
    mergersBuilder.setOutputSpec({{ "main" }, "TST", "HISTO", 0 });
    MergerConfig mergerConfig;
    mergerConfig.inputObjectTimespan = { mergersInputObjectTimespan };
    std::vector<std::pair<size_t, size_t>> param = {{mergersPublicationInterval, 1}};
    mergerConfig.publicationDecision = { mergersPublicationDecision, param };
    mergerConfig.mergedObjectTimespan = { MergedObjectTimespan::FullHistory };
    mergerConfig.topologySize = { TopologySize::NumberOfLayers, mergersLayers };
    mergersBuilder.setConfig(mergerConfig);

    mergersBuilder.generateInfrastructure(specs);

    DataProcessorSpec printer{
      "printer-bins",
      Inputs{
        { "histo", "TST", "HISTO", 0 }
      },
      Outputs{},
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback) [](InitContext&) {
          return (AlgorithmSpec::ProcessCallback) [](ProcessingContext& processingContext) mutable {
            auto histo = processingContext.inputs().get<TH1F*>("histo");
            std::string bins = "BINS:";
            for (int i = 1; i <= histo->GetNbinsX(); i++) {
              bins += " " + std::to_string((int) histo->GetBinContent(i));
              if (i >= 100) {
                LOG(info) << "Trimming the output to 100 entries, total is: " << histo->GetNbinsX();
                break;
              }
            }
            LOG(info) << bins;
          };
        }
      }
    };
    specs.push_back(printer);
  }

  return specs;
}
// clang-format on

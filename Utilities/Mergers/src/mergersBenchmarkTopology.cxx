// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file    mergersBenchmarkTopology.cxx
/// \author  Piotr Konopka
///
/// \brief This is a DPL workflow to benchmark Mergers

#include "Mergers/MergerBuilder.h"

#include <Framework/CompletionPolicy.h>

#include <TH1F.h>
#include <memory>
#include <random>

using namespace o2::framework;
using namespace o2::experimental::mergers;

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
  options.push_back({"mergers-merge-decision", VariantType::String, "publication", {"At which occasion objects are merged: 'arrival' or 'publication'"}});
  options.push_back({"mergers-publication-decision", VariantType::String, "interval", {"When merged objects are published: interval or all-updated"}});
  options.push_back({"mergers-publication-interval", VariantType::Double, 10.0, {"Publication interval of merged object [s]. It takes effect with --mergers-publication-decision interval"}});
  options.push_back(
    {"mergers-ownership-mode", VariantType::String, "diffs", {"Should the topology use 'diffs' or 'full' objects"}});
}

#include <Framework/runDataProcessing.h>
#include <fairmq/FairMQLogger.h>
#include <Mergers/MergeInterfaceOverrideExample.h>

#include "Mergers/MergerInfrastructureBuilder.h"

using namespace std::chrono;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  int objectsBins = config.options().get<int>("obj-bins");
  double objectsRate = config.options().get<double>("obj-rate");
  int objectsProducers = config.options().get<int>("obj-producers");

  int mergersLayers = config.options().get<int>("mergers-layers");
  MergingTime mergersMergeDecision =
    config.options().get<std::string>("mergers-merge-decision") == "publication" ? MergingTime::BeforePublication : MergingTime::AfterArrival;
  PublicationDecision mergersPublicationDecision =
    config.options().get<std::string>("mergers-publication-decision") == "all-updated" ? PublicationDecision::WhenXInputsUpdated : PublicationDecision::EachNSeconds;
  double mergersPublicationInterval = config.options().get<double>("mergers-publication-interval");
  OwnershipMode mergersOwnershipMode =
    config.options().get<std::string>("mergers-ownership-mode") == "full" ? OwnershipMode::Full : OwnershipMode::Integral;

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

              TH1F* histo = new TH1F("gauss", "gauss", objectsBins, -3, 3);
              histo->FillRandom("gaus", 1000);

              processingContext.outputs().adopt(
                Output{ "TST", "HISTO", static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1) }, histo);
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
    mergerConfig.ownershipMode = { mergersOwnershipMode };
    mergerConfig.publicationDecision = { mergersPublicationDecision, mergersPublicationDecision == PublicationDecision::EachNSeconds ? mergersPublicationInterval : 1.0 };
    mergerConfig.mergingTime = { mergersMergeDecision };
    mergerConfig.timespan = { Timespan::FullHistory };
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
//            LOG(INFO) << "printer invoked";
            auto histo = processingContext.inputs().get<TH1F*>("histo");
            std::string bins = "BINS:";
            for (int i = 1; i <= histo->GetNbinsX(); i++) {
              bins += " " + std::to_string((int) histo->GetBinContent(i));
            }
            LOG(INFO) << bins;
          };
        }
      }
    };
    specs.push_back(printer);
  }

  return specs;
}
// clang-format on

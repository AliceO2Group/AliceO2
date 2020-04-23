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
/// \file    multinodeBenchmarkMergers.cxx
/// \author  Piotr Konopka
///
/// \brief This is a DPL workflow to run Mergers and an input proxy for benchmarks

#include "Mergers/MergerBuilder.h"

#include <Framework/CompletionPolicy.h>

#include <memory>
#include <random>

using namespace o2::framework;
using namespace o2::mergers;

void customize(std::vector<CompletionPolicy>& policies)
{
  MergerBuilder::customizeInfrastructure(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& options)
{
  options.push_back({"mergers-layers", VariantType::Int, 1, {"Number of layers in the merger topology"}});
  options.push_back({"mergers-merge-decision", VariantType::String, "publication", {"At which occasion objects are merged: 'arrival' or 'publication'"}});
  options.push_back({"mergers-publication-decision", VariantType::String, "interval", {"When merged objects are published: interval or all-updated"}});
  options.push_back({"mergers-publication-interval", VariantType::Double, 10.0, {"Publication interval of merged object [s]. It takes effect with --mergers-publication-decision interval"}});
  options.push_back(
    {"mergers-ownership-mode", VariantType::String, "diffs", {"Should the topology use 'diffs' or 'full' objects"}});
  options.push_back({"input-channel-config", VariantType::String, "", {"Proxy input FMQ channel configuration"}});
}

#include "Framework/runDataProcessing.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <fairmq/FairMQLogger.h>
#include <TH1.h>

#include "Mergers/MergerInfrastructureBuilder.h"

using namespace std::chrono;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  int mergersLayers = config.options().get<int>("mergers-layers");
  PublicationDecision mergersPublicationDecision = PublicationDecision::EachNSeconds;
  double mergersPublicationInterval = config.options().get<double>("mergers-publication-interval");
  InputObjectsTimespan mergersOwnershipMode =
    config.options().get<std::string>("mergers-ownership-mode") == "full" ? InputObjectsTimespan::FullHistory : InputObjectsTimespan::LastDifference;
  std::string inputChannelConfig = config.options().get<std::string>("input-channel-config");

  WorkflowSpec specs;

  specs.emplace_back(std::move(specifyExternalFairMQDeviceProxy(
    "histo",
    {{{"histo"}, {"TST", "HISTO"}}},
    inputChannelConfig.c_str(),
    dplModelAdaptor())));

  MergerInfrastructureBuilder mergersBuilder;
  mergersBuilder.setInfrastructureName("histos");
  mergersBuilder.setInputSpecs({{"histo", {"TST", "HISTO"}}});
  mergersBuilder.setOutputSpec({{"main"}, "TST", "FULLHISTO", 0});
  MergerConfig mergerConfig;
  mergerConfig.inputObjectTimespan = {mergersOwnershipMode};
  mergerConfig.publicationDecision = {mergersPublicationDecision, mergersPublicationDecision == PublicationDecision::EachNSeconds ? mergersPublicationInterval : 1.0};
  mergerConfig.mergedObjectTimespan = {MergedObjectTimespan::FullHistory};
  mergerConfig.topologySize = {TopologySize::NumberOfLayers, mergersLayers};
  mergersBuilder.setConfig(mergerConfig);

  mergersBuilder.generateInfrastructure(specs);

  // clang-format off
  DataProcessorSpec printer{
    "printer-bins",
    Inputs{
      { "histo", "TST", "FULLHISTO", 0 }
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
              LOG(INFO) << "Trimming the output to 100 entries, total is: " << histo->GetNbinsX();
              break;
            }
          }
          LOG(INFO) << bins;
        };
      }
    }
  };
  specs.push_back(printer);


  return specs;
}
// clang-format on

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
/// \file    mergersTopologyExample.cxx
/// \author  Piotr Konopka
///
/// \brief This is a DPL workflow to see Mergers in action

#include "Framework/RootSerializationSupport.h"
#include "Mergers/MergerBuilder.h"

#include <Framework/CompletionPolicy.h>
#include <Framework/CompletionPolicyHelpers.h>

using namespace o2::framework;
using namespace o2::mergers;

void customize(std::vector<CompletionPolicy>& policies)
{
  MergerBuilder::customizeInfrastructure(policies);
  policies.emplace_back(CompletionPolicyHelpers::consumeWhenAny("printer-custom"));
}

#include "Framework/runDataProcessing.h"
#include "Mergers/MergerInfrastructureBuilder.h"
#include "Mergers/CustomMergeableObject.h"
#include "Framework/Logger.h"

#include <TH1F.h>
#include <memory>
#include <random>

using namespace std::chrono;

// clang-format off
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec specs;

  // one 1D histo
  {
//    WorkflowSpec specs; // enable comment to disable the workflow

    size_t producersAmount = 8;
    Inputs mergersInputs;
    for (size_t p = 0; p < producersAmount; p++) {
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
        AlgorithmSpec{(AlgorithmSpec::ProcessCallback)
                      [ p, producersAmount, srand(p) ](ProcessingContext & processingContext) mutable {

                        //            usleep(100000 + (rand() % 10000) - 5000);
                        usleep(100000);

            static int i = 0;
            if (i++ >= 1000) { return; }

            auto subspec = static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1);
            TH1F& histo = processingContext.outputs().make<TH1F>(Output{ "TST", "HISTO", subspec });
            histo.Fill(p / (double) producersAmount);
          }
        }
      };
      specs.push_back(producer);
    }

    MergerInfrastructureBuilder mergersBuilder;
    mergersBuilder.setInfrastructureName("histos");
    mergersBuilder.setInputSpecs(mergersInputs);
    mergersBuilder.setOutputSpec({{ "main" }, "TST", "HISTO", 0 });
    MergerConfig config;
    config.inputObjectTimespan = { InputObjectsTimespan::LastDifference };
    std::vector<std::pair<size_t, size_t>> param = {{5, 1}};
    config.publicationDecision = {PublicationDecision::EachNSeconds, param};
    config.mergedObjectTimespan = { MergedObjectTimespan::FullHistory };
    config.topologySize = { TopologySize::NumberOfLayers, 2 };
    mergersBuilder.setConfig(config);

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
//            LOG(info) << "printer invoked";
            auto histo = processingContext.inputs().get<TH1F*>("histo");
            std::string bins = "BINS:";
            for (int i = 1; i <= histo->GetNbinsX(); i++) {
              bins += " " + std::to_string((int) histo->GetBinContent(i));
            }
            LOG(info) << bins;
          };
        }
      }
    };
    specs.push_back(printer);
  }

  // custom merge
  {
//    WorkflowSpec specs; // enable comment to disable the workflow

    size_t producersAmount = 4;
    Inputs mergersInputs;
    for (size_t p = 0; p < producersAmount; p++) {
      mergersInputs.push_back({ "mo",               "TST",
                                "CUSTOM",           static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1),
                                Lifetime::Timeframe });
      DataProcessorSpec producer{ "producer-custom" + std::to_string(p), Inputs{},
                                  Outputs{ { { "mo" },
                                             "TST",
                                             "CUSTOM",
                                             static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1),
                                             Lifetime::Timeframe } },
                                  AlgorithmSpec{(AlgorithmSpec::ProcessCallback)[ p, producersAmount, srand(p) ](
                                    ProcessingContext & processingContext) mutable { usleep(100000);

            static int i = 0;
            if (i++ >= 1000) { return; }

            auto histo = std::make_unique<CustomMergeableObject>(1);
            auto subspec = static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1);
            processingContext.outputs().snapshot(OutputRef{ "mo", subspec }, *histo);
          }
        }
      };
      specs.push_back(producer);
    }

    MergerInfrastructureBuilder mergersBuilder;
    mergersBuilder.setInfrastructureName("custom");
    mergersBuilder.setInputSpecs(mergersInputs);
    mergersBuilder.setOutputSpec({{ "main" }, "TST", "CUSTOM", 0 });
    mergersBuilder.setOutputSpecMovingWindow({{ "main" }, "TST", "CUSTOM_MW", 0 });
    MergerConfig config;
    config.inputObjectTimespan = { InputObjectsTimespan::LastDifference };
    std::vector<std::pair<size_t, size_t>> param = {{5, 1}};
    config.publicationDecision = {PublicationDecision::EachNSeconds, param};
    config.mergedObjectTimespan = { MergedObjectTimespan::FullHistory };
    config.topologySize = { TopologySize::NumberOfLayers, 2 };
    config.publishMovingWindow = { PublishMovingWindow::Yes };
    mergersBuilder.setConfig(config);

    mergersBuilder.generateInfrastructure(specs);

    DataProcessorSpec printer{
      "printer-custom",
      Inputs{
        { "custom", "TST", "CUSTOM", 0, Lifetime::Sporadic },
        { "custom_mw", "TST", "CUSTOM_MW", 0, Lifetime::Sporadic },
      },
      Outputs{},
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback) [](InitContext&) {
          return (AlgorithmSpec::ProcessCallback) [](ProcessingContext& processingContext) mutable {
            if (processingContext.inputs().isValid("custom")) {
              auto obj = processingContext.inputs().get<CustomMergeableObject*>("custom");
              LOG(info) << "SECRET:" << obj->getSecret();
            }
            if (processingContext.inputs().isValid("custom_mw")) {
              auto mw = processingContext.inputs().get<CustomMergeableObject*>("custom_mw");
              LOG(info) << "SECRET MW:" << mw->getSecret();
            }
          };
        }
      }
    };
    specs.push_back(printer);
  }

  return specs;
}
// clang-format on

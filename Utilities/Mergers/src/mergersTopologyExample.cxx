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
/// \file    mergersTopologyExample.cxx
/// \author  Piotr Konopka
///
/// \brief This is a DPL workflow to see Mergers in action

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

#include <Framework/runDataProcessing.h>
#include <fairmq/FairMQLogger.h>
#include <Mergers/MergeInterfaceOverrideExample.h>

#include "Mergers/MergerInfrastructureBuilder.h"

using namespace std::chrono;

// clang-format off
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec specs;

  // one 1D histo, binwise
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
    config.ownershipMode = { OwnershipMode::Integral };
    config.publicationDecision = { PublicationDecision::EachNSeconds, 5 };
    config.mergingTime = { MergingTime::BeforePublication };
    config.timespan = { Timespan::FullHistory };
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


  // concatenation test
  {
//    WorkflowSpec specs; // enable comment to disable the workflow
    size_t producersAmount = 4;
    Inputs mergersInputs;
    for (size_t p = 0; p < producersAmount; p++) {
      mergersInputs.push_back({ "mo",               "TST",
                                "STRING",           static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1),
                                Lifetime::Timeframe });
      DataProcessorSpec producer{ "producer-str" + std::to_string(p), Inputs{},
                                  Outputs{ { { "mo" },
                                             "TST",
                                             "STRING",
                                             static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1),
                                             Lifetime::Timeframe } },
                                  AlgorithmSpec{(AlgorithmSpec::ProcessCallback)
                                                [p, producersAmount](ProcessingContext& processingContext) mutable {

            usleep(1000000);
            char str[2] = "a";
            str[0] += p;
            auto subspec = static_cast<o2::header::DataHeader::SubSpecificationType>(p + 1);
            // we make a framework-owned object with a title, but we do not modify it
            processingContext.outputs().make<TObjString>(Output{ "TST", "STRING", subspec }, str);
                                                } } };
      specs.push_back(producer);
    }

    MergerInfrastructureBuilder mergersBuilder;
    mergersBuilder.setInfrastructureName("strings");
    mergersBuilder.setInputSpecs(mergersInputs);
    mergersBuilder.setOutputSpec({{ "main" }, "TST", "STRING", 0 });
    MergerConfig config;
    config.ownershipMode = { OwnershipMode::Full };
    config.mergingMode = { MergingMode::Concatenate };
    config.publicationDecision = { PublicationDecision::WhenXInputsUpdated, 1 };
    config.mergingTime = { MergingTime::BeforePublication };
    config.topologySize = { TopologySize::NumberOfLayers, 2 };
    mergersBuilder.setConfig(config);

    mergersBuilder.generateInfrastructure(specs);

    DataProcessorSpec printer{
      "printer-collections",
      Inputs{
        { "string", "TST", "STRING", 0 }
      },
      Outputs{},
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback) [](InitContext&) {
          return (AlgorithmSpec::ProcessCallback) [](ProcessingContext& processingContext) mutable {
//            LOG(INFO) << "printer invoked";
            auto stringArray = processingContext.inputs().get<TObjArray*>("string");
            std::string full = "FULL: ";
            for (const auto& obj : *stringArray) {
              auto str = dynamic_cast<TObjString*>(obj);

              full += str->GetString();
            }
            LOG(INFO) << full;
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

            auto histo = std::make_unique<MergeInterfaceOverrideExample>(1);
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
    MergerConfig config;
    config.ownershipMode = { OwnershipMode::Integral };
    config.publicationDecision = { PublicationDecision::EachNSeconds, 5 };
    config.mergingTime = { MergingTime::BeforePublication };
    config.timespan = { Timespan::FullHistory };
    config.topologySize = { TopologySize::NumberOfLayers, 1 };
    mergersBuilder.setConfig(config);

    mergersBuilder.generateInfrastructure(specs);

    DataProcessorSpec printer{
      "printer-custom",
      Inputs{
        { "custom", "TST", "CUSTOM", 0 }
      },
      Outputs{},
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback) [](InitContext&) {
          return (AlgorithmSpec::ProcessCallback) [](ProcessingContext& processingContext) mutable {
            auto obj = processingContext.inputs().get<MergeInterfaceOverrideExample*>("custom");
            LOG(INFO) << "SECRET:" << obj->getSecret();
          };
        }
      }
    };
    specs.push_back(printer);
  }

  return specs;
}
// clang-format on

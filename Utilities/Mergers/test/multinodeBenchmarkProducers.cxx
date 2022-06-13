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
/// \file    multinodeBenchmarkProducers.cxx
/// \author  Piotr Konopka
///
/// \brief This is a DPL workflow with TH1 producers used to benchmark Mergers

#include <Framework/ConfigParamSpec.h>
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& options)
{
  options.push_back({"obj-bins", VariantType::Int, 100, {"Number of bins in a histogram"}});
  options.push_back({"obj-rate", VariantType::Double, 1.0, {"Number of objects per second sent by one producer"}});
  options.push_back({"obj-producers", VariantType::Int, 4, {"Number of objects producers"}});
  options.push_back({"obj-per-message", VariantType::Int, 1, {"Number objects per message (in one TCollection)"}});
  options.push_back({"output-channel-config", VariantType::String, "", {"Proxy output FMQ channel configuration"}});
  options.push_back({"first-subspec", VariantType::Int, 1, {"First subSpec of the parallel producers, the rest will be incremental"}});
}

#include <Framework/runDataProcessing.h>
#include "Framework/Logger.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <TRandomGen.h>
#include <TObjArray.h>
#include <TH1F.h>

using namespace std::chrono;
using SubSpec = o2::header::DataHeader::SubSpecificationType;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  int objectsBins = config.options().get<int>("obj-bins");
  double objectsRate = config.options().get<double>("obj-rate");
  int objectsProducers = config.options().get<int>("obj-producers");
  int objectsPerMessage = config.options().get<int>("obj-per-message");
  std::string outputChannelConfig = config.options().get<std::string>("output-channel-config");
  SubSpec subSpec = static_cast<SubSpec>(config.options().get<int>("first-subspec"));
  WorkflowSpec specs;
  // clang-format off
  // one 1D histo
  for (size_t p = 0; p < objectsProducers; p++, subSpec++) {
    DataProcessorSpec producer{
      "producer-histo" + std::to_string(subSpec),
      Inputs{},
      Outputs{ { { "histo" }, "TST", "HISTO", subSpec} },
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback)[=](InitContext& ictx) {
          const size_t randoms = 10000;
          int periodus = static_cast<int>(1000000 / objectsRate);
          TRandom gen;
          gen.SetSeed(p);

          double randomsArray[randoms];
          TObjArray* collection = new TObjArray();
          collection->SetOwner(true);

          for (size_t i = 0; i < objectsPerMessage; i++) {
            TH1I* h = new TH1I(std::to_string(i).c_str(), "uni", objectsBins, 0, 1);
            collection->Add(h);
          }

          return (AlgorithmSpec::ProcessCallback)[=](ProcessingContext& pctx) mutable {

            static auto lastTime = steady_clock::now() - std::chrono::microseconds(periodus * p / objectsProducers);
            auto now = steady_clock::now();
            if (duration_cast<microseconds>(now - lastTime).count() > periodus) {
              lastTime += microseconds(periodus);

              if (objectsPerMessage > 1) {
                for (auto o : *collection) {
                  gen.RndmArray(randoms, randomsArray);

                  TH1I* h = dynamic_cast<TH1I*>(o);
                  h->Reset();
                  h->FillN(randoms, randomsArray, nullptr);
                }

                collection->SetOwner(false);
                pctx.outputs().snapshot({"TST", "HISTO", subSpec}, *collection);
                collection->SetOwner(true);
              } else {
                gen.RndmArray(randoms, randomsArray);

                TH1I* h = dynamic_cast<TH1I*>(collection->At(0));
                h->Reset();
                h->FillN(randoms, randomsArray, nullptr);

                pctx.outputs().snapshot({"TST", "HISTO", subSpec}, *h);
              }
            }
          };
        }
      }
    };
    specs.push_back(producer);

    // We spawn one proxy per each producer to simulate the real scenario.
    specs.emplace_back(
      std::move(
        specifyFairMQDeviceOutputProxy(
          ("histo-proxy-" + std::to_string(p)).c_str(),
          {{"histo", "TST", "HISTO", subSpec }}, outputChannelConfig.c_str())));
  }

  return specs;
}
// clang-format on

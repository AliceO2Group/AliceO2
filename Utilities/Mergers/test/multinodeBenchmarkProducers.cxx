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
/// \file    multinodeBenchmarkProducers.cxx
/// \author  Piotr Konopka
///
/// \brief This is a DPL workflow with TH1 producers used to benchmark Mergers

#include <Framework/ConfigParamSpec.h>

#include <TH1F.h>
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& options)
{
  options.push_back({"obj-bins", VariantType::Int, 100, {"Number of bins in a histogram"}});
  options.push_back({"obj-rate", VariantType::Double, 1.0, {"Number of objects per second sent by one producer"}});
  options.push_back({"obj-producers", VariantType::Int, 4, {"Number of objects producers"}});
  options.push_back({"output-channel-config", VariantType::String, "", {"Proxy output FMQ channel configuration"}});
  options.push_back({"first-subspec", VariantType::Int, 1, {"First subSpec of the parallel producers, the rest will be incremental"}});
}

#include <Framework/runDataProcessing.h>
#include <fairmq/FairMQLogger.h>
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <TRandomGen.h>

using namespace std::chrono;
using SubSpec = o2::header::DataHeader::SubSpecificationType;

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  int objectsBins = config.options().get<int>("obj-bins");
  double objectsRate = config.options().get<double>("obj-rate");
  int objectsProducers = config.options().get<int>("obj-producers");
  std::string outputChannelConfig = config.options().get<std::string>("output-channel-config");
  SubSpec subSpec = static_cast<SubSpec>(config.options().get<int>("first-subspec"));
  WorkflowSpec specs;
  // clang-format off
  // one 1D histo
  for (size_t p = 0; p < objectsProducers; p++) {
    DataProcessorSpec producer{
      "producer-histo" + std::to_string(subSpec),
      Inputs{},
      Outputs{ { { "histo" }, "TST", "HISTO", subSpec} },
      AlgorithmSpec{
        (AlgorithmSpec::InitCallback)[=](InitContext& ictx) {
          int periodus = static_cast<int>(1000000 / objectsRate);
          TRandomMT64 gen;
          gen.SetSeed(p);

          return (AlgorithmSpec::ProcessCallback)[=](ProcessingContext& pctx) mutable {

            static auto lastTime = steady_clock::now();
            auto now = steady_clock::now();

            const size_t randoms = 10000;

            if (duration_cast<microseconds>(now - lastTime).count() > periodus) {

              lastTime += microseconds(periodus);

              TH1F& histo = pctx.outputs().make<TH1F>({ "TST", "HISTO", subSpec }, "uni", "uni", objectsBins, 0, 1000);
              for (size_t i = 0; i < randoms; i++) {
                histo.Fill(gen.Rndm() * 1000);
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


    subSpec++;
  }

  return specs;
}
// clang-format on

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

#include "Framework/DataProcessorSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "DataGeneratorSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  policies.push_back(o2::framework::CallbacksPolicy{
    [](o2::framework::DeviceSpec const& spec, o2::framework::ConfigContext const& context) -> bool {
      return spec.name == "calib-tf-dispatcher"; // apply policy only to the upstream device
    },
    [](o2::framework::CallbackService& service, o2::framework::InitContext& context) {
      const auto& hbfu = o2::raw::HBFUtils::Instance();
      long startTime = hbfu.startTime > 0 ? hbfu.startTime : std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
      service.set<o2::framework::CallbackService::Id::NewTimeslice>(
        [startTime](o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph) {
          const auto& hbfu = o2::raw::HBFUtils::Instance();
          dh.firstTForbit = hbfu.getFirstIRofTF({0, hbfu.orbitFirstSampled}).orbit + int64_t(hbfu.nHBFPerTF) * dh.tfCounter;
          dh.runNumber = hbfu.runNumber;
          dph.creation = startTime + (dh.firstTForbit - hbfu.orbitFirst) * o2::constants::lhc::LHCOrbitMUS * 1.e-3;
        });
    }});
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"lanes", o2::framework::VariantType::Int, 2, {"number of data generator lanes"}});
  workflowOptions.push_back(ConfigParamSpec{"gen-norm", o2::framework::VariantType::Int, 1, {"nominal number of expected generators"}});
  workflowOptions.push_back(ConfigParamSpec{"gen-slot", o2::framework::VariantType::Int, 0, {"generate TFs of slot in [0 : gen-norm) range"}});
  workflowOptions.push_back(ConfigParamSpec{"pressure", o2::framework::VariantType::Float, 1.f, {"generation / processing rate factor"}});
  workflowOptions.push_back(ConfigParamSpec{"mean-latency", o2::framework::VariantType::Int, 1000, {"mean latency of the processor in microseconds"}});
  workflowOptions.push_back(ConfigParamSpec{"latency-spread", o2::framework::VariantType::Int, 100, {"latency gaussian RMS of the processor in microseconds"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto nlanes = std::max(1, configcontext.options().get<int>("lanes"));
  auto ngen = std::max(1, configcontext.options().get<int>("gen-norm"));
  auto slot = std::max(0, configcontext.options().get<int>("gen-slot"));
  auto latency = std::max(1, configcontext.options().get<int>("mean-latency"));
  auto latencyRMS = std::max(1, configcontext.options().get<int>("latency-spread"));
  auto pressure = std::max(0.001f, configcontext.options().get<float>("pressure"));
  if (slot >= ngen) {
    slot = 0;
    ngen = 1;
  }
  specs.emplace_back(getTFDispatcherSpec(slot, ngen, nlanes, std::max(1, int(float(latency) / nlanes / pressure))));
  specs.emplace_back(timePipeline(getTFProcessorCalibInfoTOFSpec(latency, latencyRMS), nlanes));
  return specs;
}

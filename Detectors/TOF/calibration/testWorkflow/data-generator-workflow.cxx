// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessorSpec.h"
#include "DataGeneratorSpec.h"

using namespace o2::framework;

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
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
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
  specs.emplace_back(timePipeline(getTFProcessorSpec(latency, latencyRMS), nlanes));
  return specs;
}

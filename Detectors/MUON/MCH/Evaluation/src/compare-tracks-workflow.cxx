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

#include "CommonUtils/ConfigurableParam.h"
#include "CompareTask.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/ConfigContext.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"
#include <string>

using namespace o2::framework;
using namespace o2::mch;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  Inputs inputs{};
  inputs.emplace_back("rofs1", "MCH", "TRACKROFS", 0, Lifetime::Timeframe);
  inputs.emplace_back("tracks1", "MCH", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusters1", "MCH", "TRACKCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("rofs2", "MCH", "TRACKROFS", 1, Lifetime::Timeframe);
  inputs.emplace_back("tracks2", "MCH", "TRACKS", 1, Lifetime::Timeframe);
  inputs.emplace_back("clusters2", "MCH", "TRACKCLUSTERS", 1, Lifetime::Timeframe);

  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                             // orbitResetTime
                                                                false,                             // GRPECS=true
                                                                false,                             // GRPLHCIF
                                                                true,                              // GRPMagField
                                                                false,                             // askMatLUT
                                                                o2::base::GRPGeomRequest::Aligned, // geometry
                                                                inputs);
  specs.emplace_back(DataProcessorSpec{
    "mch-compare-tracks",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<eval::CompareTask>(ccdbRequest)},
    Options{
      {"precision", VariantType::Double, 1.e-4, {"precision used for comparisons"}},
      {"apply-track-selection", VariantType::Bool, false, {"apply standard track selection cuts"}},
      {"print-diff", VariantType::Bool, false, {"print differences"}},
      {"print-all", VariantType::Bool, false, {"print all differences"}},
      {"outfile", VariantType::String, "compare.root", {"output Root filename"}},
      {"pdf-outfile", VariantType::String, "", {"output PDF filename (leave empty for no PDF output)"}}}});

  return specs;
}

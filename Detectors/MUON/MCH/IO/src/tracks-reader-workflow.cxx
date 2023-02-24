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

#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "MCHIO/TrackReaderSpec.h"
#include <vector>

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"enable-mc", VariantType::Bool, false, ConfigParamSpec::HelpString{"Propagate MC info"}},
    {"digits", VariantType::Bool, false, ConfigParamSpec::HelpString{"Read associated digits"}},
    {"subspec", VariantType::UInt32, static_cast<uint32_t>(0), ConfigParamSpec::HelpString{"SubSpec of outputs (only needed in workflows reading different mchtracks.root files)"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& config)
{
  bool useMC = config.options().get<bool>("enable-mc");
  bool digits = config.options().get<bool>("digits");
  uint32_t subspec = config.options().get<uint32_t>("subspec");
  WorkflowSpec specs{o2::mch::getTrackReaderSpec(useMC, "mch-tracks-reader", digits, subspec)};
  o2::raw::HBFUtilsInitializer hbfIni(config, specs);
  return specs;
}

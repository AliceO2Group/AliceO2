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

/// \file   MCH/Workflow/src/sim-digits-reader-workflow.cxx
/// \brief  MCH digits reader workflow
/// \author Michael Winn <Michael.Winn at cern.ch>
/// \date   17 April 2021

#include <vector>
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigParamSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DigitReaderSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"disable-mc", VariantType::Bool, false, {"Do not propagate MC info"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  workflowOptions.insert(workflowOptions.end(), options.begin(), options.end());
  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  bool useMC = !cfgc.options().get<bool>("disable-mc");

  WorkflowSpec specs;
  specs.emplace_back(o2::mch::getDigitReaderSpec(useMC));

  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return specs;
}

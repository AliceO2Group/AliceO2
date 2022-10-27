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

/// \file   MID/Workflow/src/efficiency-workflow.cxx
/// \brief  Workflow to compute the MID chamber efficiency
/// \author Livia Terlizzi <Livia.Terlizzi at cern.ch>
/// \date   20 September 2022

#include <string>
#include <vector>
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "MIDWorkflow/EfficiencySpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.emplace_back("configKeyValues", VariantType::String, "",
                               ConfigParamSpec::HelpString{"Semicolon separated key=value strings"});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  WorkflowSpec effspecs{};

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  effspecs.emplace_back(o2::mid::getEfficiencySpec());

  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2mideff-workflow_configuration.ini");

  return effspecs;
}
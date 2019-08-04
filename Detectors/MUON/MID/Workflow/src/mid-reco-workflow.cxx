// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   mid-reco-workflow.cxx
/// \brief  MID reconstruction workflow
/// \author Diego Stocco <diego.stocco at cern.ch>
/// \date   12 June 2019

#include <string>
#include <vector>
#include "Framework/Variant.h"
#include "SimConfig/ConfigurableParam.h"
#include "MIDWorkflow/RecoWorkflow.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{
    "disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}});

  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {keyvaluehelp}});
}

// ------------------------------------------------------------------

#include "Framework/ConfigParamSpec.h"
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the digitizer workflow
  o2::conf::ConfigurableParam::writeINI("o2mid-recoflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");

  return std::move(o2::mid::getRecoWorkflow(useMC));
}

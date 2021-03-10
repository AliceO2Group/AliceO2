// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HMPIDWorkflow/DigitReaderSpec.h"
#include "HMPIDWorkflow/ClusterizerSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
// ------------------------------------------------------------------

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{
    "disable-mc", VariantType::Bool, false, {"disable MC propagation even if available"}});
  workflowOptions.push_back(ConfigParamSpec{
    "disable-root-input", VariantType::Bool, false, {"disable root-files input readers"}});
  workflowOptions.push_back(ConfigParamSpec{
    "disable-root-output", VariantType::Bool, false, {"disable root-files output writers"}});
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {keyvaluehelp}});
}

#include "Framework/runDataProcessing.h" // the main driver

/// The standalone workflow executable for HMPID reconstruction workflow
/// - digit reader
/// - clusterer

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  auto useMC = !cfgc.options().get<bool>("disable-mc");
  auto disableRootInp = cfgc.options().get<bool>("disable-root-input");
  auto disableRootOut = cfgc.options().get<bool>("disable-root-output");

  if (!disableRootInp) {
    specs.emplace_back(o2::hmpid::getDigitReaderSpec(useMC));
  }
  specs.emplace_back(o2::hmpid::getHMPIDClusterizerSpec(useMC));
  if (!disableRootOut) {
    // RS Here cluster writer should go
  }

  return std::move(specs);
}

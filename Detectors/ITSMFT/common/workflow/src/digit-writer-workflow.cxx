// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITSMFTWorkflow/DigitWriterSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"writeHW", VariantType::Bool, false, {"write hardware info"}},
    ConfigParamSpec{"disable-mc", VariantType::Bool, false, {"disable mc truth"}},
    ConfigParamSpec{"runmft", VariantType::Bool, false, {"expect MFT data"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  bool useMC = !cfgc.options().get<bool>("disable-mc");
  bool writeHW = cfgc.options().get<bool>("writeHW");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  if (cfgc.options().get<bool>("runmft")) {
    wf.emplace_back(o2::itsmft::getMFTDigitWriterSpec(useMC, writeHW));
  } else {
    wf.emplace_back(o2::itsmft::getITSDigitWriterSpec(useMC));
  }
  return wf;
}

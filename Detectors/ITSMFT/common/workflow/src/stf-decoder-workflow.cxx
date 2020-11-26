// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ITSMFTWorkflow/STFDecoderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"writeHW", VariantType::Bool, false, {"write hardware informations of chips (def : false)"}},
    ConfigParamSpec{"runmft", VariantType::Bool, false, {"source detector is MFT (default ITS)"}},
    ConfigParamSpec{"no-clusters", VariantType::Bool, false, {"do not produce clusters (def: produce)"}},
    ConfigParamSpec{"no-cluster-patterns", VariantType::Bool, false, {"do not produce clusters patterns (def: produce)"}},
    ConfigParamSpec{"digits", VariantType::Bool, false, {"produce digits (def: skip)"}},
    ConfigParamSpec{"dict-file", VariantType::String, "", {"name of the cluster-topology dictionary file"}},
    ConfigParamSpec{"noise-file", VariantType::String, "", {"name of the noise map file"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  auto writeHW = cfgc.options().get<bool>("writeHW");
  auto doClusters = !cfgc.options().get<bool>("no-clusters");
  auto doPatterns = doClusters && !cfgc.options().get<bool>("no-cluster-patterns");
  auto doDigits = cfgc.options().get<bool>("digits");
  auto dict = cfgc.options().get<std::string>("dict-file");
  auto noise = cfgc.options().get<std::string>("noise-file");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  if (cfgc.options().get<bool>("runmft")) {
    wf.emplace_back(o2::itsmft::getSTFDecoderMFTSpec(doClusters, doPatterns, doDigits, dict, noise, writeHW));
  } else {
    wf.emplace_back(o2::itsmft::getSTFDecoderITSSpec(doClusters, doPatterns, doDigits, dict, noise));
  }
  return wf;
}

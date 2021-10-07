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
    ConfigParamSpec{"runmft", VariantType::Bool, false, {"source detector is MFT (default ITS)"}},
    ConfigParamSpec{"no-clusters", VariantType::Bool, false, {"do not produce clusters (def: produce)"}},
    ConfigParamSpec{"no-cluster-patterns", VariantType::Bool, false, {"do not produce clusters patterns (def: produce)"}},
    ConfigParamSpec{"digits", VariantType::Bool, false, {"produce digits (def: skip)"}},
    ConfigParamSpec{"enable-calib-data", VariantType::Bool, false, {"produce GBT calibration stream (def: skip)"}},
    ConfigParamSpec{"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    ConfigParamSpec{"dataspec", VariantType::String, "", {"selection string for the input data, if not provided <DET>Raw:<DET>/RAWDATA with DET=ITS or MFT will be used"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  o2::itsmft::STFDecoderInp inp;
  inp.doClusters = !cfgc.options().get<bool>("no-clusters");
  inp.doPatterns = inp.doClusters && !cfgc.options().get<bool>("no-cluster-patterns");
  inp.doDigits = cfgc.options().get<bool>("digits");
  inp.doCalib = cfgc.options().get<bool>("enable-calib-data");
  inp.askSTFDist = !cfgc.options().get<bool>("ignore-dist-stf");
  inp.inputSpec = cfgc.options().get<std::string>("dataspec");
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  if (cfgc.options().get<bool>("runmft")) {
    if (inp.inputSpec.empty()) {
      inp.inputSpec = "mftRAW:MFT/RAWDATA";
    }
    inp.origin = o2::header::gDataOriginMFT;
    inp.deviceName = "mft-stf-decoder";
  } else {
    if (inp.inputSpec.empty()) {
      inp.inputSpec = "itsRAW:ITS/RAWDATA";
    }
    inp.origin = o2::header::gDataOriginITS;
    inp.deviceName = "its-stf-decoder";
  }
  wf.emplace_back(o2::itsmft::getSTFDecoderSpec(inp));

  return wf;
}

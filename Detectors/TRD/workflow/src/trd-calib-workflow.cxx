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

#include "Framework/DataProcessorSpec.h"
#include "TRDWorkflowIO/TRDCalibVdExBReaderSpec.h"
#include "TRDWorkflowIO/TRDCalibGainReaderSpec.h"
#include "TRDWorkflow/VdAndExBCalibSpec.h"
#include "TRDWorkflow/GainCalibSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"vdexb", o2::framework::VariantType::Bool, false, {"enable VDrift and ExB calibration"}},
    {"enable-root-input-vdexb", o2::framework::VariantType::Bool, false, {"enable root-files input readers for vdexb"}},
    {"gain", o2::framework::VariantType::Bool, false, {"enable gain calibration"}},
    {"enable-root-input-gain", o2::framework::VariantType::Bool, false, {"enable root-files input readers for gain"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto enableRootInpVDEXB = configcontext.options().get<bool>("enable-root-input-vdexb");
  auto enableRootInpGain = configcontext.options().get<bool>("enable-root-input-gain");
  auto enableVDEXB = configcontext.options().get<bool>("vdexb");
  auto enableGain = configcontext.options().get<bool>("gain");
  WorkflowSpec specs;
  if (enableVDEXB) {
    if (enableRootInpVDEXB) {
      specs.emplace_back(o2::trd::getTRDCalibVdExBReaderSpec());
    }
    specs.emplace_back(getTRDVdAndExBCalibSpec());
  }
  if (enableGain) {
    if (enableRootInpGain) {
      specs.emplace_back(o2::trd::getTRDCalibGainReaderSpec());
    }
    specs.emplace_back(getTRDGainCalibSpec());
  }
  if (specs.empty()) {
    LOG(warn) << "No Calibration Mode selected, dilly-dallying...";
  }
  return specs;
}

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
#include "TRDWorkflowIO/TRDCalibReaderSpec.h"
#include "TRDWorkflowIO/TRDDigitReaderSpec.h"
#include "TRDWorkflow/VdAndExBCalibSpec.h"
#include "TRDWorkflow/NoiseCalibSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"enable-root-input", o2::framework::VariantType::Bool, false, {"enable root-files input readers"}},
    {"vDriftAndExB", o2::framework::VariantType::Bool, false, {"enable vDrift and ExB calibration"}},
    {"noise", o2::framework::VariantType::Bool, false, {"enable noise and pad status calibration"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto enableRootInp = configcontext.options().get<bool>("enable-root-input");

  WorkflowSpec specs;

  if (configcontext.options().get<bool>("vDriftAndExB")) {
    if (enableRootInp) {
      specs.emplace_back(o2::trd::getTRDCalibReaderSpec());
    }
    specs.emplace_back(getTRDVdAndExBCalibSpec());
  }

  if (configcontext.options().get<bool>("noise")) {
    if (enableRootInp) {
      specs.emplace_back(o2::trd::getTRDDigitReaderSpec(false));
      specs.emplace_back(o2::trd::getTRDNoiseCalibSpec());
    }
    // specs.emplace_back(o2::trd::getTRDNoiseCalibAggregatorSpec());
  }

  return specs;
}

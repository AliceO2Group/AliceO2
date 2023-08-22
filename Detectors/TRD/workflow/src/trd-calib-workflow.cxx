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
#include "TRDWorkflowIO/TRDPHReaderSpec.h"
#include "TRDWorkflow/VdAndExBCalibSpec.h"
#include "TRDWorkflow/GainCalibSpec.h"
#include "TRDWorkflow/NoiseCalibSpec.h"
#include "TRDWorkflow/T0FitSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"enable-root-input", o2::framework::VariantType::Bool, false, {"enable root-files input readers"}},
    {"vDriftAndExB", o2::framework::VariantType::Bool, false, {"enable vDrift and ExB calibration"}},
    {"noise", o2::framework::VariantType::Bool, false, {"enable noise and pad status calibration"}},
    {"gain", o2::framework::VariantType::Bool, false, {"enable gain calibration"}},
    {"t0", o2::framework::VariantType::Bool, false, {"enable t0 fit"}},
    {"calib-dds-collection-index", VariantType::Int, -1, {"allow only single collection to produce calibration objects (use -1 for no limit)"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
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
    }
    int ddsCollectionIdx = configcontext.options().get<int>("calib-dds-collection-index");
    bool noiseCalibIsDummy = true;
    if (ddsCollectionIdx != -1) {
      char* colIdx = getenv("DDS_COLLECTION_INDEX");
      int myIdx = colIdx ? atoi(colIdx) : -1;
      if (myIdx == ddsCollectionIdx) {
        LOG(info) << "TRD noise calib is enabled for this collection, my index " << myIdx;
        noiseCalibIsDummy = false;
      } else {
        LOG(info) << "TRD noise calib is disabled for this collection, my index " << myIdx;
      }
    } else {
      noiseCalibIsDummy = false;
    }
    specs.emplace_back(o2::trd::getTRDNoiseCalibSpec(noiseCalibIsDummy));
  }

  if (configcontext.options().get<bool>("gain")) {
    specs.emplace_back(getTRDGainCalibSpec());
  }

  if (configcontext.options().get<bool>("t0")) {
    if (enableRootInp) {
      specs.emplace_back(o2::trd::getTRDPHReaderSpec());
    }
    specs.emplace_back(getTRDT0FitSpec());
  }

  return specs;
}

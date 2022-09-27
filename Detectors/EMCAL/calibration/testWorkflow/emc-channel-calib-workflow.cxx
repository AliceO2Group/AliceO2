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

/// @file   emc-channel-calib-workflow.cxx
/// @author Hannah Bossi
/// @since  2020-12-01
/// @brief  Basic workflow for EMCAL bad channel calibration (adapted from tof-calib-workflow.cxx)

#include "EMCALChannelCalibratorSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/Variant.h"
#include "Framework/ConfigParamSpec.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

using namespace o2::framework;
using namespace o2::emcal;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"calibType", VariantType::String, "time", {"choose which calibration should be performed: time for tiem calibration, badchannel for bad channel calibration"}},
    {"no-loadCalibParamsFromCCDB", VariantType::Bool, false, {"disabled by default such that calib params are taken from the ccdb. If enabled, calib params are taken from EMCALCalibParams.h directly"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  std::string calibType = cfgc.options().get<std::string>("calibType");
  bool loadCalibParamsFromCCDB = !cfgc.options().get<bool>("no-loadCalibParamsFromCCDB");

  WorkflowSpec specs;
  specs.emplace_back(getEMCALChannelCalibDeviceSpec(calibType, loadCalibParamsFromCCDB));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  // o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);
  return specs;
}
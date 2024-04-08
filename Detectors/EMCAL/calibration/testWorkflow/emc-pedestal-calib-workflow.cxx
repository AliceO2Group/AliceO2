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
/// @author Joshua Koenig
/// @since  2024-03-25
/// @brief  Basic workflow for EMCAL pedestal calibration

#include "CommonUtils/ConfigurableParam.h"
#include "EMCALCalibration/PedestalCalibDevice.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

#include <string>

using namespace o2::framework;
using namespace o2::emcal;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"dumpToFile", VariantType::Bool, false, {"if output (that goes to DCS ccdb) should be stored in txt file for local debugging"}},
    {"addRunNumber", VariantType::Bool, false, {"if true, run number will be added to ccdb file as first element of the string"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  bool dumpToFile = cfgc.options().get<bool>("dumpToFile");
  bool addRunNumber = cfgc.options().get<bool>("addRunNumber");

  WorkflowSpec specs;
  specs.emplace_back(o2::emcal::getPedestalCalibDevice(dumpToFile, addRunNumber));

  return specs;
}

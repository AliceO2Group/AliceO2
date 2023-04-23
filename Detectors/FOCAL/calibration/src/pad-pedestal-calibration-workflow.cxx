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

#include <iostream>
#include "FOCALCalibration/PadPedestalCalibDevice.h"
#include "Framework/DataProcessorSpec.h"
#include "CommonUtils/ConfigurableParam.h"

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb cpv calibration objects"}});
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"debug", o2::framework::VariantType::Bool, false, {"debug mode (store calibration objects to local file)"}});
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"path", o2::framework::VariantType::String, "./", {"path to store temp files"}});
  workflowOptions.push_back(o2::framework::ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings"}});
}

#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  o2::framework::WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto useCCDB = configcontext.options().get<bool>("use-ccdb");
  auto debugMode = configcontext.options().get<bool>("debug");
  auto path = configcontext.options().get<std::string>("path");

  specs.emplace_back(o2::focal::getPadPedestalCalibDevice(useCCDB, path, debugMode));
  return specs;
}
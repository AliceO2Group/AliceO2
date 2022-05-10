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

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "MCHDigitFiltering/DigitFilteringSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2;
using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"input-digits-data-description", VariantType::String, "DIGITS", {"description string for the input digits data"}});
  workflowOptions.push_back(ConfigParamSpec{"output-digits-data-description", VariantType::String, "F-DIGITS", {"description string for the output digits data"}});
  workflowOptions.push_back(ConfigParamSpec{"input-digitrofs-data-description", VariantType::String, "DIGITROFS", {"description string for the input digit rofs data"}});
  workflowOptions.push_back(ConfigParamSpec{"output-digitrofs-data-description", VariantType::String, "F-DIGITROFS", {"description string for the output digit rofs data"}});
  workflowOptions.push_back(ConfigParamSpec{"input-digitlabels-data-description", VariantType::String, "DIGITLABELS", {"description string for the input digit labels data (not used if disable-mc is true)"}});
  workflowOptions.push_back(ConfigParamSpec{"output-digitlabels-data-description", VariantType::String, "F-DIGITLABELS", {"description string for the output digit labels data (not used if disable-mc is true)"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", VariantType::Bool, false, {"Do not propagate MC info"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  o2::conf::ConfigurableParam::updateFromString(cc.options().get<std::string>("configKeyValues"));

  WorkflowSpec wf;

  wf.emplace_back(o2::mch::getDigitFilteringSpec(
    not cc.options().get<bool>("disable-mc"),
    "mch-digits-filtering",
    cc.options().get<std::string>("input-digits-data-description"),
    cc.options().get<std::string>("output-digits-data-description"),
    cc.options().get<std::string>("input-digitrofs-data-description"),
    cc.options().get<std::string>("output-digitrofs-data-description"),
    cc.options().get<std::string>("input-digitlabels-data-description"),
    cc.options().get<std::string>("output-digitlabels-data-description")));

  return wf;
}

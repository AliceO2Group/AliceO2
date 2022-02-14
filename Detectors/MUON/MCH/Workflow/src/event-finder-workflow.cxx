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

/// \file event-finder-workflow.cxx
/// \brief Implementation of a DPL device to group MCH digits based on MID information
///
/// \author Philippe Pillot, Subatech

#include <string>
#include <vector>
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/WorkflowSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "EventFinderSpec.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"input-digits-data-description", VariantType::String, "F-DIGITS", {"description string for the input MCH digits data"}});
  workflowOptions.push_back(ConfigParamSpec{"output-digits-data-description", VariantType::String, "E-F-DIGITS", {"description string for the output MCH digits data"}});
  workflowOptions.push_back(ConfigParamSpec{"input-digitrofs-data-description", VariantType::String, "F-DIGITROFS", {"description string for the input MCH digit rofs data"}});
  workflowOptions.push_back(ConfigParamSpec{"output-digitrofs-data-description", VariantType::String, "E-F-DIGITROFS", {"description string for the output MCH digit rofs data"}});
  workflowOptions.push_back(ConfigParamSpec{"input-digitlabels-data-description", VariantType::String, "F-DIGITLABELS", {"description string for the input MCH digit labels data (not used if disable-mc is true)"}});
  workflowOptions.push_back(ConfigParamSpec{"output-digitlabels-data-description", VariantType::String, "E-F-DIGITLABELS", {"description string for the output MCH digit labels data (not used if disable-mc is true)"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", VariantType::Bool, false, {"Do not propagate MC info"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  o2::conf::ConfigurableParam::updateFromString(cc.options().get<std::string>("configKeyValues"));
  return WorkflowSpec{o2::mch::getEventFinderSpec(
    not cc.options().get<bool>("disable-mc"),
    "mch-event-finder",
    cc.options().get<std::string>("input-digits-data-description"),
    cc.options().get<std::string>("output-digits-data-description"),
    cc.options().get<std::string>("input-digitrofs-data-description"),
    cc.options().get<std::string>("output-digitrofs-data-description"),
    cc.options().get<std::string>("input-digitlabels-data-description"),
    cc.options().get<std::string>("output-digitlabels-data-description"))};
}

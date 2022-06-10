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

/// \file digits-to-preclusters-workflow.cxx
/// \brief This is an executable that runs the preclusterization via DPL.
///
/// This is an executable that takes digits from the Data Processing Layer, runs the preclusterization and sends the preclusters via the Data Processing Layer.
///
/// \author Philippe Pillot, Subatech
/// \author Andrea Ferrero, CEA

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "CommonUtils/ConfigurableParam.h"
#include "MCHPreClustering/PreClusterFinderSpec.h"

using namespace o2;
using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back(ConfigParamSpec{"input-digitrofs-data-description", VariantType::String, "TC-F-DIGITROFS", {"description string for the input ROF data"}});
  workflowOptions.emplace_back(ConfigParamSpec{"input-digits-data-description", VariantType::String, "F-DIGITS", {"description string for the input digits data"}});
  workflowOptions.emplace_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& cc)
{
  o2::conf::ConfigurableParam::updateFromString(cc.options().get<std::string>("configKeyValues"));
  return {
    o2::mch::getPreClusterFinderSpec(
      "mch-preclustering",
      cc.options().get<std::string>("input-digits-data-description"),
      cc.options().get<std::string>("input-digitrofs-data-description"))};
}

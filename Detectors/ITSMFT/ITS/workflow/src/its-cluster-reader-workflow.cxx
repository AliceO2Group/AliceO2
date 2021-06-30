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

#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "with-mc",
      o2::framework::VariantType::Bool,
      false,
      {"propagate MC labels"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "without-patterns",
      o2::framework::VariantType::Bool,
      false,
      {"do not propagate pixel patterns"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "configKeyValues",
      VariantType::String,
      "",
      {"Semicolon separated key=value strings"}});
}

#include "Framework/runDataProcessing.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cc)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cc.options().get<std::string>("configKeyValues"));
  auto withMC = cc.options().get<bool>("with-mc");
  auto withPatterns = !cc.options().get<bool>("without-patterns");

  specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(withMC, withPatterns));

  return specs;
}

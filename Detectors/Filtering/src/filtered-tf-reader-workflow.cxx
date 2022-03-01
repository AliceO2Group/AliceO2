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
#include "Framework/CallbacksPolicy.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"disable-mc", VariantType::Bool, false, {"disable mc truth"}},
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"
#include "FilteredTFReaderSpec.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cc)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cc.options().get<std::string>("configKeyValues"));
  auto useMC = !cc.options().get<bool>("disable-mc");

  specs.emplace_back(o2::filtering::getFilteredTFReaderSpec(useMC));

  return specs;
}

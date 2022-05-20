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

#include "MCHGeometryTransformer/ClusterTransformerSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigContext.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Variant.h"
#include "Framework/WorkflowSpec.h"

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.emplace_back("configKeyValues",
                               o2::framework::VariantType::String, "",
                               o2::framework::ConfigParamSpec::HelpString{"Semicolon separated key=value strings"});
  workflowOptions.emplace_back("mch-disable-geometry-from-ccdb",
                               o2::framework::VariantType::Bool, false,
                               o2::framework::ConfigParamSpec::HelpString{"do not read geometry from ccdb"});
}

#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(const o2::framework::ConfigContext& configContext)
{
  o2::conf::ConfigurableParam::updateFromString(configContext.options().get<std::string>("configKeyValues"));
  bool disableCcdb = configContext.options().get<bool>("mch-disable-geometry-from-ccdb");

  return o2::framework::WorkflowSpec{o2::mch::getClusterTransformerSpec("mch-cluster-transformer", disableCcdb)};
}

// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "FITWorkflow/FITMergeIntegrateClusterSpec.h"
#include "DataFormatsFT0/RecPoints.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
  };
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec wf;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  wf.emplace_back(o2::fit::getFITMergeIntegrateClusterSpec<o2::ft0::RecPoints>());
  return wf;
}

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/ConfigParamSpec.h"
#include "DataSampling/DataSampling.h"
#include "Framework/CompletionPolicyHelpers.h"
#include <vector>

using namespace o2::framework;
using namespace o2::utilities;

void customize(std::vector<CompletionPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

void customize(std::vector<ChannelConfigurationPolicy>& policies)
{
  DataSampling::CustomizeInfrastructure(policies);
}

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"config", VariantType::String, "", {"path to the Data Sampling configuration file"}});
  workflowOptions.push_back(ConfigParamSpec{"dispatchers", VariantType::Int, 1, {"amount of parallel Dispatchers"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  auto configurationPath = config.options().get<std::string>("config");
  auto numberOfDispatchers = config.options().get<int>("dispatchers");

  WorkflowSpec specs;
  DataSampling::GenerateInfrastructure(specs, configurationPath, numberOfDispatchers);
  return specs;
}
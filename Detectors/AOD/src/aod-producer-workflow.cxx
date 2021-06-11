// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AODProducerWorkflow/AODProducerWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"ignore-aod-writer", VariantType::Int, 0, {"Ignore DPL AOD writer and write tables directly into a file. 0 -- off, != 0 -- on"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  //  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("ignore-aod-writer"));
  int ignoreWriter = configcontext.options().get<int>("ignore-aod-writer");
  return std::move(o2::aodproducer::getAODProducerWorkflow(ignoreWriter));
}

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonUtils/ConfigurableParam.h"
#include "centralEventFilterProcessor.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  workflowOptions.push_back(ConfigParamSpec{"config", o2::framework::VariantType::String, "train_config.json", {"Configuration of the filtering"}});
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto config = configcontext.options().get<std::string>("config");

  if (config.empty()) {
    LOG(FATAL) << "We need a configuration for the centralEventFilterProcessor";
    throw std::runtime_error("incompatible options provided");
  }

  WorkflowSpec specs;
  specs.emplace_back(o2::aod::filtering::getCentralEventFilterProcessorSpec(config));
  return std::move(specs);
}

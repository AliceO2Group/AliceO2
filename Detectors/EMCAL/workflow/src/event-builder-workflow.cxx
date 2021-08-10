// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction

#include <string>
#include <sstream>
#include <vector>
#include "EMCALWorkflow/EventBuilderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/Logger.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options{ConfigParamSpec{"subspecifications", VariantType::String, "", {"Comma-separated list of subspecifications"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  std::vector<unsigned int> specs;
  std::stringstream parser(cfgc.options().get<std::string>("subspecifications"));
  std::string buffer;
  while (std::getline(parser, buffer, ',')) {
    specs.emplace_back(std::stoi(buffer));
  }
  if (!specs.size()) {
    LOG(FATAL) << "No input specs for defined";
  }
  WorkflowSpec wf;
  wf.emplace_back(o2::emcal::getEventBuilderSpec(specs));
  return wf;
}

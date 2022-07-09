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
#include "Framework/CallbacksPolicy.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"data-origin", o2::framework::VariantType::String, "NIL", {"ouput data origin"}});
  workflowOptions.push_back(ConfigParamSpec{"subspec", o2::framework::VariantType::Int, 0, {"ouput subspec"}});
  workflowOptions.push_back(ConfigParamSpec{"device-name", o2::framework::VariantType::String, "irframe-reader", {"device name"}});
  workflowOptions.push_back(ConfigParamSpec{"file-name", o2::framework::VariantType::String, "o2_irframe.root", {"default input file name"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}});
  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions);
}

#include "Framework/runDataProcessing.h"
#include "GlobalTrackingWorkflowReaders/IRFrameReaderSpec.h"

WorkflowSpec defineDataProcessing(ConfigContext const& cc)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cc.options().get<std::string>("configKeyValues"));
  o2::header::DataOrigin origin;
  origin.runtimeInit(cc.options().get<std::string>("data-origin").c_str());

  specs.emplace_back(o2::globaltracking::getIRFrameReaderSpec(origin, (uint32_t)cc.options().get<int>("subspec"),
                                                              cc.options().get<std::string>("device-name"), cc.options().get<std::string>("file-name")));

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cc, specs);

  return specs;
}

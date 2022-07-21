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

#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsRaw/DistSTFSenderSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/Logger.h"
#include <string>

using namespace o2::framework;
using namespace o2::raw;

//_________________________________________________________
void customize(std::vector<CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

//_________________________________________________________
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}},
    {"max-tf", o2::framework::VariantType::Int, 1, {"how many TFs to process"}},
    {"dist-tf-subspec", o2::framework::VariantType::Int, 0xccdb, {"Subspec to use for FLP/DISTSUBTIMEFRAME"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

//_________________________________________________________
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  int maxTF = configcontext.options().get<int>("max-tf");
  unsigned subSpec = static_cast<unsigned>(configcontext.options().get<int>("dist-tf-subspec"));
  specs.push_back(o2::raw::getDistSTFSenderSpec(maxTF > 0 ? maxTF : 1, subSpec));
  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);
  return specs;
}

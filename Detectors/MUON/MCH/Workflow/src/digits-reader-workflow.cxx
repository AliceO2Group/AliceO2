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

/// \file digits-reader-workflow.cxx
/// \brief Implementation of a DPL device to read digits from a binary file
///
/// \author Philippe Pillot, Subatech
/// \author Andrea Ferrero, CEA

#include "Framework/runDataProcessing.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "DigitSamplerSpec.h"

using namespace o2::framework;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(const ConfigContext& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  WorkflowSpec wf{o2::mch::getDigitSamplerSpec()};

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, wf);

  return std::move(wf);
}

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

/// \file reco-reader-workflow.cxx
/// \brief Implementation of ZDC reco reader
///
/// \author ruben,shahoian@cern.ch, pietro.cortese@cern.ch

#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "ZDCWorkflow/RecoReaderSpec.h"
#include "CommonUtils/ConfigurableParam.h"

using namespace o2::framework;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::string keyvaluehelp("Semicolon separated key=value strings ...");
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {keyvaluehelp}});
  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  DataProcessorSpec producer = o2::zdc::getRecoReaderSpec();
  specs.push_back(producer);

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return specs;
}

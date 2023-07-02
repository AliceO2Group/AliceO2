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

#include "GlobalTrackingWorkflow/StrangenessTrackingSpec.h"
#include "GlobalTrackingWorkflow/StrangenessTrackingWriterSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "StrangenessTracking/StrangenessTrackingConfigParam.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflow/SecondaryVertexingSpec.h"
#include "GlobalTrackingWorkflow/SecondaryVertexWriterSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/PrimaryVertexReaderSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/ConfigContext.h"
#include <vector>

using namespace o2::framework;
using namespace o2::strangeness_tracking;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  o2::raw::HBFUtilsInitializer::addConfigOption(options);
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto useRootInput = !configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");

  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  o2::conf::ConfigurableParam::writeINI("o2strangeness_tracking_workflow_configuration.ini");
  GID::mask_t itsSource = GID::getSourceMask(GID::ITS); // ITS tracks and clusters

  WorkflowSpec specs;
  specs.emplace_back(o2::strangeness_tracking::getStrangenessTrackerSpec(itsSource, useMC));
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, itsSource, itsSource, itsSource, useMC, itsSource);
  o2::globaltracking::InputHelper::addInputSpecsPVertex(configcontext, specs, useMC); // P-vertex is always needed
  o2::globaltracking::InputHelper::addInputSpecsSVertex(configcontext, specs);        // S-vertex is always needed

  if (!disableRootOut) {
    specs.emplace_back(getStrangenessTrackingWriterSpec(useMC));
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  // write the configuration used for the reco workflow

  return std::move(specs);
}

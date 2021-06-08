// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDWorkflow/TRDTrackingWorkflow.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"Disable MC labels"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}},
    {"tracking-sources", VariantType::String, std::string{o2::dataformats::GlobalTrackID::ALL}, {"comma-separated list of sources to use for tracking"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::dataformats::GlobalTrackID::mask_t allowedSources = o2::dataformats::GlobalTrackID::getSourcesMask("ITS-TPC,TPC");
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2trdtracking-workflow_configuration.ini");
  auto disableRootInp = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  o2::dataformats::GlobalTrackID::mask_t srcTRD = allowedSources & o2::dataformats::GlobalTrackID::getSourcesMask(configcontext.options().get<std::string>("tracking-sources"));

  auto wf = o2::trd::getTRDTrackingWorkflow(disableRootInp, disableRootOut, srcTRD);

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, wf);

  return std::move(wf);
}

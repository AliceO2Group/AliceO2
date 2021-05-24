// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCWorkflow/ClusterSharingMapSpec.h"
#include "GlobalTrackingWorkflow/TPCITSMatchingSpec.h"
#include "GlobalTrackingWorkflow/TrackWriterTPCITSSpec.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"use-ft0", o2::framework::VariantType::Bool, false, {"use FT0 in matching"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, "ITS,TPC", {"comma-separated list of sources to use: ITS,TPC,TPC-TOF,TPC-TRD,TPC-TRD-TOF"}},
    {"produce-calibration-data", o2::framework::VariantType::Bool, false, {"produce output for TPC vdrift calibration"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

// the matcher process requires the TPC sector completion to trigger and data on
// all defined routes
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // the TPC sector completion policy checks when the set of TPC/CLUSTERNATIVE data is complete
  // in addition we require to have input from all other routes
  policies.push_back(o2::tpc::TPCSectorCompletionPolicy("itstpc-track-matcher",
                                                        o2::tpc::TPCSectorCompletionPolicy::Config::RequireAll,
                                                        o2::framework::InputSpec{"cluster", o2::framework::ConcreteDataTypeMatcher{"TPC", "CLUSTERNATIVE"}})());
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2matchtpcits-workflow_configuration.ini");

  GID::mask_t alowedSources = GID::getSourcesMask("ITS,TPC,TPC-TOF");
  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  auto useFT0 = configcontext.options().get<bool>("use-ft0");
  if (useFT0) {
    src |= GID::getSourceMask(GID::FT0);
  }
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto calib = configcontext.options().get<bool>("produce-calibration-data");

  LOG(INFO) << "Data sources: " << GID::getSourcesNames(src);
  auto srcL = src | GID::getSourcesMask("ITS,TPC"); // ITS is neadded always, TPC must be loaded even if bare TPC tracks are not used in matching

  o2::framework::WorkflowSpec specs;
  specs.emplace_back(o2::globaltracking::getTPCITSMatchingSpec(srcL, useFT0, calib, !GID::includesSource(GID::TPC, src), useMC));

  if (!configcontext.options().get<bool>("disable-root-output")) {
    specs.emplace_back(o2::globaltracking::getTrackWriterTPCITSSpec(useMC));
  }

  // the only clusters MC which is need with useMC is ITS (for afterburner), for the rest we use tracks MC labels
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcL, srcL, srcL,
                                                 useMC, GID::getSourceMask(GID::ITS));

  return std::move(specs);
}

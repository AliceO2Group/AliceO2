// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "GlobalTrackingWorkflow/PrimaryVertexWriterSpec.h"
#include "GlobalTrackingWorkflow/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/VertexTrackMatcherSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/TrackReaderSpec.h"
#include "FT0Workflow/RecPointReaderSpec.h"

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"validate-with-ft0", o2::framework::VariantType::Bool, false, {"use FT0 time for vertex validation"}},
    {"disable-vertex-track-matching", o2::framework::VariantType::Bool, false, {"disable matching of vertex to non-contributor tracks"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2primary-vertexing-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootInp = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto validateWithFT0 = configcontext.options().get<bool>("validate-with-ft0");
  auto disableMatching = configcontext.options().get<bool>("disable-vertex-track-matching");

  WorkflowSpec specs;
  if (!disableRootInp) {
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
    if (validateWithFT0) {
      specs.emplace_back(o2::ft0::getRecPointReaderSpec(false));
    }
  }
  specs.emplace_back(o2::vertexing::getPrimaryVertexingSpec(validateWithFT0, useMC));

  if (!disableMatching && !disableRootInp) {
    specs.emplace_back(o2::its::getITSTrackReaderSpec(false));
    specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(false));
    specs.emplace_back(o2::vertexing::getVertexTrackMatcherSpec());
  }

  if (!disableRootOut) {
    specs.emplace_back(o2::vertexing::getPrimaryVertexWriterSpec(disableMatching, useMC));
  }
  return std::move(specs);
}

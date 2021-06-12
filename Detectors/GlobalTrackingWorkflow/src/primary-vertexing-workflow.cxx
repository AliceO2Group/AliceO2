// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GlobalTrackingWorkflowHelpers/InputHelper.h"
#include "GlobalTrackingWorkflow/PrimaryVertexingSpec.h"
#include "GlobalTrackingWorkflow/PrimaryVertexWriterSpec.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/VertexTrackMatcherSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "FT0Workflow/RecPointReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;
using GID = o2::dataformats::GlobalTrackID;
using DetID = o2::detectors::DetID;
// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"vertexing-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use in vertexing"}},
    {"validate-with-ft0", o2::framework::VariantType::Bool, false, {"use FT0 time for vertex validation"}},
    {"vetex-track-matching-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use in vertex-track associations or \"none\" to disable matching"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;

  GID::mask_t allowedSourcesPV = GID::getSourcesMask("ITS,ITS-TPC,ITS-TPC-TRD,ITS-TPC-TOF");
  GID::mask_t allowedSourcesVT = GID::getSourcesMask("ITS,MFT,TPC,ITS-TPC,,TPC-TOF,TPC-TRD,ITS-TPC-TRD,ITS-TPC-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2primary-vertexing-workflow_configuration.ini");


  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto validateWithFT0 = configcontext.options().get<bool>("validate-with-ft0");

  GID::mask_t srcPV = allowedSourcesPV & GID::getSourcesMask(configcontext.options().get<std::string>("vertexing-sources"));
  GID::mask_t srcVT = allowedSourcesVT & GID::getSourcesMask(configcontext.options().get<std::string>("vetex-track-matching-sources"));
  if (validateWithFT0) {
    srcPV |= GID::getSourceMask(GID::FT0);
  }
  GID::mask_t srcComb = srcPV | srcVT;
  GID::mask_t dummy, srcClus = GID::includesDet(DetID::TOF, srcComb) ? GID::getSourceMask(GID::TOF) : dummy;

  specs.emplace_back(o2::vertexing::getPrimaryVertexingSpec(srcPV, validateWithFT0, useMC));
  if (!srcVT.none()) {
    specs.emplace_back(o2::vertexing::getVertexTrackMatcherSpec(srcVT));
  }

  // only TOF clusters are needed if TOF is involved, no clusters MC needed
  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, srcClus, srcComb, srcComb, useMC, dummy);

  if (!disableRootOut) {
    specs.emplace_back(o2::vertexing::getPrimaryVertexWriterSpec(srcVT.none(), useMC));
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}

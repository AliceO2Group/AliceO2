// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "GlobalTrackingWorkflow/SecondaryVertexingSpec.h"
#include "GlobalTrackingWorkflow/SecondaryVertexWriterSpec.h"
#include "GlobalTrackingWorkflow/TrackTPCITSReaderSpec.h"
#include "GlobalTrackingWorkflow/PrimaryVertexReaderSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCWorkflow/TrackReaderSpec.h"
#include "TOFWorkflow/TOFMatchedReaderSpec.h"
#include "TOFWorkflowUtils/ClusterReaderSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
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
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"vertexing-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use in vertexing"}},
    {"enable-cascade-finder", o2::framework::VariantType::Bool, false, {"run cascade finder for each V0"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  GID::mask_t alowedSourcesSV = GID::getSourcesMask("ITS,ITS-TPC,ITS-TPC-TOF,TPC-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2secondary-vertexing-workflow_configuration.ini");
  bool useMC = false;
  auto disableRootInp = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto enableCasc = configcontext.options().get<bool>("enable-cascade-finder");

  GID::mask_t srcSV = alowedSourcesSV & GID::getSourcesMask(configcontext.options().get<std::string>("vertexing-sources"));
  WorkflowSpec specs;
  if (!disableRootInp) {
    if (srcSV[GID::ITS]) {
      specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
    }
    if (srcSV[GID::TPC]) {
      specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));
    }
    if (srcSV[GID::ITSTPC] || srcSV[GID::ITSTPCTOF]) { // ITSTPCTOF does not provide tracks, only matchInfo
      specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
    }
    if (srcSV[GID::ITSTPCTOF]) {
      specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(useMC, false, false)); // MC, MatchInfo_glo, no TOF_TPCtracks
      specs.emplace_back(o2::tof::getClusterReaderSpec(false));                  // RSTODO Needed just to set the time of ITSTPC track, consider moving to MatchInfoTOF
    }
    if (srcSV[GID::TPCTOF]) {
      specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(useMC, true, true)); // mc, MatchInfo_TPC, TOF_TPCtracks
    }
    specs.emplace_back(o2::vertexing::getPrimaryVertexReaderSpec(useMC));
  }

  specs.emplace_back(o2::vertexing::getSecondaryVertexingSpec(srcSV, enableCasc));

  if (!disableRootOut) {
    specs.emplace_back(o2::vertexing::getSecondaryVertexWriterSpec());
  }
  return std::move(specs);
}

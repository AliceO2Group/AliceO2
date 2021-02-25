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
#include "TOFWorkflow/TOFMatchedReaderSpec.h"
#include "TOFWorkflowUtils/ClusterReaderSpec.h"
#include "FT0Workflow/RecPointReaderSpec.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/ConfigParamSpec.h"

using namespace o2::framework;
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
    {"onlyDet", VariantType::String, std::string{DetID::NONE}, {"comma-separated list of detectors to use. Overrides skipDet"}},
    {"skipDet", VariantType::String, std::string{DetID::NONE}, {"comma-separate list of detectors to skip"}},
    {"validate-with-ft0", o2::framework::VariantType::Bool, false, {"use FT0 time for vertex validation"}},
    {"disable-vertex-track-matching", o2::framework::VariantType::Bool, false, {"disable matching of vertex to non-contributor tracks"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  DetID::mask_t dets = DetID::getMask("ITS,TPC,TRD,TOF,FT0");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2primary-vertexing-workflow_configuration.ini");

  if (!configcontext.helpOnCommandLine()) {
    auto mskOnly = DetID::getMask(configcontext.options().get<std::string>("onlyDet"));
    auto mskSkip = DetID::getMask(configcontext.options().get<std::string>("skipDet"));
    if (mskOnly.any()) {
      dets &= mskOnly;
    } else {
      dets ^= mskSkip;
    }
  }

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootInp = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto validateWithFT0 = configcontext.options().get<bool>("validate-with-ft0");
  auto disableMatching = configcontext.options().get<bool>("disable-vertex-track-matching");

  bool readerTrackITSDone = false, readerTrackITSTPCDone = false, readerGloTOFDone = false;

  if (!disableRootInp) {
    if (dets[DetID::ITS]) {
      specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
      readerTrackITSDone = true;
    }
    if (dets[DetID::TPC]) {
      specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
      readerTrackITSTPCDone = true;
      if (dets[DetID::TRD]) {
        // RSTODO will add once TRD tracking available
      }
      if (dets[DetID::TOF]) {
        specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(true, false, false)); // MC, MatchInfo_glo, no TOF_TPCtracks
        readerGloTOFDone = true;
        specs.emplace_back(o2::tof::getClusterReaderSpec(false)); // RSTODO Needed just to set the time of ITSTPC track, consider moving to MatchInfoTOF
      }
    }
    if (validateWithFT0) {
      specs.emplace_back(o2::ft0::getRecPointReaderSpec(false));
    }
  }
  specs.emplace_back(o2::vertexing::getPrimaryVertexingSpec(dets, validateWithFT0, useMC));

  if (!disableMatching) {
    if (!disableRootInp) {

      if (dets[DetID::ITS]) {
        if (!readerTrackITSDone) {
          specs.emplace_back(o2::its::getITSTrackReaderSpec(false));
        }
      }
      if (dets[DetID::TPC]) {
        if (dets[DetID::ITS] && !readerTrackITSTPCDone) {
          specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(false));
        }
        specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(false));
        if (dets[DetID::TOF]) {
          if (!readerGloTOFDone) {
            specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(false, false, false)); // MC, MatchInfo_glo, no TOF_TPCtracks
            specs.emplace_back(o2::tof::getClusterReaderSpec(false));                  // RSTODO Needed just to set the time of ITSTPC track, consider moving to MatchInfoTOF
          }
          specs.emplace_back(o2::tof::getTOFMatchedReaderSpec(false, true, true)); // mc, MatchInfo_TPC, TOF_TPCtracks
        }
      }
    }
    specs.emplace_back(o2::vertexing::getVertexTrackMatcherSpec(dets));
  }

  if (!disableRootOut) {
    specs.emplace_back(o2::vertexing::getPrimaryVertexWriterSpec(disableMatching, useMC));
  }
  return std::move(specs);
}

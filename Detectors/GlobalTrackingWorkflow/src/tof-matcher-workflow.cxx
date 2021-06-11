// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicy.h"
#include "TPCReaderWorkflow/TPCSectorCompletionPolicy.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedReaderSpec.h"
#include "GlobalTrackingWorkflow/TOFMatcherSpec.h"
#include "TOFWorkflowIO/TOFMatchedWriterSpec.h"
#include "TOFWorkflowIO/TOFCalibWriterSpec.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "GlobalTrackingWorkflowHelpers/InputHelper.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;
using GID = o2::dataformats::GlobalTrackID;
// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable MC propagation even if available"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input reader"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writer"}},
    {"track-sources", VariantType::String, std::string{GID::ALL}, {"comma-separated list of sources to use"}},
    {"use-fit", o2::framework::VariantType::Bool, false, {"enable access to fit info for calibration"}},
    {"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}},
    {"output-type", o2::framework::VariantType::String, "matching-info,calib-info", {"matching-info, calib-info"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  WorkflowSpec specs;
  GID::mask_t alowedSources = GID::getSourcesMask("ITS,TPC,ITS-TPC,TPC-TOF,ITS-TPC-TOF");

  // Update the (declared) parameters if changed from the command line
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  // write the configuration used for the workflow
  o2::conf::ConfigurableParam::writeINI("o2match-tof-workflow_configuration.ini");

  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootIn = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOut = configcontext.options().get<bool>("disable-root-output");
  auto useFIT = configcontext.options().get<bool>("use-fit");
  auto useCCDB = configcontext.options().get<bool>("use-ccdb");
  bool writeTOFTPC = false;

  bool writematching = 0;
  bool writecalib = 0;
  auto outputType = configcontext.options().get<std::string>("output-type");
  if (outputType.rfind("matching-info") < outputType.size()) {
    writematching = 1;
  }
  if (outputType.rfind("calib-info") < outputType.size()) {
    writecalib = 1;
  }

  if (!writecalib) {
    useFIT = false;
  }

  LOG(INFO) << "TOF MATCHER WORKFLOW configuration";
  LOG(INFO) << "TOF track inputs = " << configcontext.options().get<std::string>("track-sources");
  LOG(INFO) << "TOF output = " << outputType;
  LOG(INFO) << "TOF disable-mc = " << configcontext.options().get<std::string>("disable-mc");
  LOG(INFO) << "TOF use-ccdb = " << useCCDB;
  LOG(INFO) << "TOF use-fit = " << useFIT;
  LOG(INFO) << "TOF disable-root-input = " << disableRootIn;
  LOG(INFO) << "TOF disable-root-output = " << disableRootOut;

  GID::mask_t src = alowedSources & GID::getSourcesMask(configcontext.options().get<std::string>("track-sources"));
  GID::mask_t dummy;
  GID::mask_t nonemask = GID::getSourcesMask(GID::NONE);

  o2::globaltracking::InputHelper::addInputSpecs(configcontext, specs, nonemask, nonemask, src, useMC, dummy); // only tracks needed

  if (!disableRootIn) { // input data loaded from root files
    LOG(INFO) << "Insert TOF Cluster Reader";
    specs.emplace_back(o2::tof::getClusterReaderSpec(useMC));
  }

  specs.emplace_back(o2::globaltracking::getTOFMatcherSpec(src, useMC, useFIT));

  if (!disableRootOut) {
    if (writematching) {
      specs.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC, "o2match_toftpc.root", writeTOFTPC));
    }
    if (writecalib) {
      specs.emplace_back(o2::tof::getTOFCalibWriterSpec("o2calib_toftpc.root", writeTOFTPC));
    }
  }

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(configcontext, specs);

  return std::move(specs);
}

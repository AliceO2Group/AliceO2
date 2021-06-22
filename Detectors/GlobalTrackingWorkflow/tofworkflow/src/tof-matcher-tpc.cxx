// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tof-reco-workflow.cxx
/// @author Francesco Noferini
/// @since  2019-05-22
/// @brief  Basic DPL workflow for TOF reconstruction starting from digits

#include "DetectorsBase/Propagator.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowIO/TOFMatchedWriterSpec.h"
#include "TOFWorkflowIO/TOFCalibWriterSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "TOFWorkflow/RecoWorkflowWithTPCSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "FairLogger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "TPCReaderWorkflow/ClusterReaderSpec.h"
#include "TPCWorkflow/ClusterSharingMapSpec.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"

// GRP
#include "DataFormatsParameters/GRPObject.h"

// FIT
#include "FT0Workflow/RecPointReaderSpec.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"output-type", o2::framework::VariantType::String, "matching-info", {"matching-info, calib-info"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable sending of MC information, TBI"}},
    {"tof-sectors", o2::framework::VariantType::String, "0-17", {"TOF sector range, e.g. 5-7,8,9 ,TBI"}},
    {"tof-lanes", o2::framework::VariantType::Int, 1, {"number of parallel lanes up to the matcher, TBI"}},
    {"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}},
    {"use-fit", o2::framework::VariantType::Bool, false, {"enable access to fit info for calibration"}},
    {"input-desc", o2::framework::VariantType::String, "CRAWDATA", {"Input specs description string"}},
    {"tpc-refit", o2::framework::VariantType::Bool, false, {"refit matched TPC tracks"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}},
    {"cosmics", o2::framework::VariantType::Bool, false, {"reco for cosmics"}},
    {"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings ..."}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;

/// The workflow executable for the stand alone TOF reconstruction workflow
/// The basic workflow for TOF reconstruction is defined in RecoWorkflow.cxx
/// and contains the following default processors
/// - digit reader
/// - clusterer
/// - cluster raw decoder
/// - track-TOF matcher
///
/// The default workflow can be customized by specifying input and output types
/// e.g. digits, raw, clusters.
///
/// MC info is processed by default, disabled by using command line option `--disable-mc`
///
/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  // the lane configuration defines the subspecification ids to be distributed among the lanes.
  auto nLanes = cfgc.options().get<int>("tof-lanes");
  auto outputType = cfgc.options().get<std::string>("output-type");

  bool writematching = 0;
  bool writecalib = 0;

  if (outputType.rfind("matching-info") < outputType.size()) {
    writematching = 1;
  }
  if (outputType.rfind("calib-info") < outputType.size()) {
    writecalib = 1;
  }

  auto useMC = !cfgc.options().get<bool>("disable-mc");
  auto useCCDB = cfgc.options().get<bool>("use-ccdb");
  auto useFIT = cfgc.options().get<bool>("use-fit");
  auto doTPCRefit = cfgc.options().get<bool>("tpc-refit");
  bool disableRootInput = cfgc.options().get<bool>("disable-root-input");
  bool disableRootOutput = cfgc.options().get<bool>("disable-root-output");
  auto isCosmics = cfgc.options().get<bool>("cosmics");

  LOG(INFO) << "TOF RECO WORKFLOW configuration";
  LOG(INFO) << "TOF output = " << cfgc.options().get<std::string>("output-type");
  LOG(INFO) << "TOF sectors = " << cfgc.options().get<std::string>("tof-sectors");
  LOG(INFO) << "TOF disable-mc = " << cfgc.options().get<std::string>("disable-mc");
  LOG(INFO) << "TOF lanes = " << cfgc.options().get<std::string>("tof-lanes");
  LOG(INFO) << "TOF use-ccdb = " << cfgc.options().get<std::string>("use-ccdb");
  LOG(INFO) << "TOF use-fit = " << cfgc.options().get<std::string>("use-fit");
  LOG(INFO) << "TOF tpc-refit = " << cfgc.options().get<bool>("tpc-refit");
  LOG(INFO) << "TOF disable-root-input = " << disableRootInput;
  LOG(INFO) << "TOF disable-root-output = " << disableRootOutput;

  // useMC = false;
  // LOG(INFO) << "TOF disable MC forced";
  // writecalib = false;
  // LOG(INFO) << "TOF CalibInfo disabled (forced)";

  if (!disableRootInput) { // input data loaded from root files
    LOG(INFO) << "Insert TOF Cluster Reader";
    specs.emplace_back(o2::tof::getClusterReaderSpec(useMC));

    LOG(INFO) << "Insert TPC Track Reader";
    specs.emplace_back(o2::tpc::getTPCTrackReaderSpec(useMC));

    if (doTPCRefit) {
      LOG(INFO) << "Insert TPC Cluster Reader";
      specs.emplace_back(o2::tpc::getClusterReaderSpec(false));
      specs.emplace_back(o2::tpc::getClusterSharingMapSpec());
    }

    if (useFIT) {
      LOG(INFO) << "Insert FIT RecPoint Reader";
      specs.emplace_back(o2::ft0::getRecPointReaderSpec(useMC));
    }
  }

  LOG(INFO) << "Insert TOF Matching";
  specs.emplace_back(o2::tof::getTOFRecoWorkflowWithTPCSpec(useMC, useFIT, doTPCRefit, isCosmics));

  if (!disableRootOutput) {
    if (writematching) {
      LOG(INFO) << "Insert TOF Matched Info Writer";
      specs.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC, "o2match_toftpc.root", true));
    }
    if (writecalib) {
      LOG(INFO) << "Insert TOF Calib Info Writer";
      specs.emplace_back(o2::tof::getTOFCalibWriterSpec("o2calib_toftpc.root", true));
    }
  }

  LOG(INFO) << "Number of active devices = " << specs.size();

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return std::move(specs);
}

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
#include "GlobalTrackingWorkflow/TrackTPCITSReaderSpec.h"
#include "TOFWorkflowUtils/ClusterReaderSpec.h"
#include "TOFWorkflow/TOFMatchedWriterSpec.h"
#include "TOFWorkflow/TOFCalibWriterSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "TOFWorkflow/RecoWorkflowSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "FairLogger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"

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
  workflowOptions.push_back(ConfigParamSpec{"input-type", o2::framework::VariantType::String, "clusters,tracks", {"clusters, tracks, fit"}});
  workflowOptions.push_back(ConfigParamSpec{"output-type", o2::framework::VariantType::String, "matching-info,calib-info", {"matching-info, calib-info"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"disable sending of MC information, TBI"}});
  workflowOptions.push_back(ConfigParamSpec{"tof-sectors", o2::framework::VariantType::String, "0-17", {"TOF sector range, e.g. 5-7,8,9 ,TBI"}});
  workflowOptions.push_back(ConfigParamSpec{"tof-lanes", o2::framework::VariantType::Int, 1, {"number of parallel lanes up to the matcher, TBI"}});
  workflowOptions.push_back(ConfigParamSpec{"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}});
  workflowOptions.push_back(ConfigParamSpec{"use-fit", o2::framework::VariantType::Bool, false, {"enable access to fit info for calibration"}});
  workflowOptions.push_back(ConfigParamSpec{"input-desc", o2::framework::VariantType::String, "CRAWDATA", {"Input specs description string"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings ..."}});
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

  if (!cfgc.helpOnCommandLine()) {
    std::string inputGRP = o2::base::NameConf::getGRPFileName();
    o2::base::Propagator::initFieldFromGRP(inputGRP);
    const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
    if (!grp) {
      LOG(ERROR) << "This workflow needs a valid GRP file to start";
      return specs;
    }
    o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
    //  o2::conf::ConfigurableParam::writeINI("o2tofrecoflow_configuration.ini");
  }
  // the lane configuration defines the subspecification ids to be distributed among the lanes.
  // auto tofSectors = o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tof-sectors"));
  // std::vector<int> laneConfiguration = tofSectors;
  auto nLanes = cfgc.options().get<int>("tof-lanes");
  auto inputType = cfgc.options().get<std::string>("input-type");
  auto outputType = cfgc.options().get<std::string>("output-type");

  bool writematching = 0;
  bool writecalib = 0;

  if (outputType.rfind("matching-info") < outputType.size()) {
    writematching = 1;
  }
  if (outputType.rfind("calib-info") < outputType.size()) {
    writecalib = 1;
  }

  bool clusterinput = 0;
  bool trackinput = 0;
  bool fitinput = 0;

  if (inputType.rfind("clusters") < inputType.size()) {
    clusterinput = 1;
  }
  if (inputType.rfind("tracks") < inputType.size()) {
    trackinput = 1;
  }
  auto useMC = !cfgc.options().get<bool>("disable-mc");
  auto useCCDB = cfgc.options().get<bool>("use-ccdb");
  auto useFIT = cfgc.options().get<bool>("use-fit");
  bool disableRootInput = cfgc.options().get<bool>("disable-root-input");
  bool disableRootOutput = cfgc.options().get<bool>("disable-root-output");

  if (inputType.rfind("fit") < inputType.size()) {
    fitinput = 1;
    useFIT = 1;
  }

  LOG(INFO) << "TOF RECO WORKFLOW configuration";
  LOG(INFO) << "TOF input = " << cfgc.options().get<std::string>("input-type");
  LOG(INFO) << "TOF output = " << cfgc.options().get<std::string>("output-type");
  LOG(INFO) << "TOF sectors = " << cfgc.options().get<std::string>("tof-sectors");
  LOG(INFO) << "TOF disable-mc = " << cfgc.options().get<std::string>("disable-mc");
  LOG(INFO) << "TOF lanes = " << cfgc.options().get<std::string>("tof-lanes");
  LOG(INFO) << "TOF use-ccdb = " << cfgc.options().get<std::string>("use-ccdb");
  LOG(INFO) << "TOF use-fit = " << cfgc.options().get<std::string>("use-fit");
  LOG(INFO) << "TOF disable-root-input = " << disableRootInput;
  LOG(INFO) << "TOF disable-root-output = " << disableRootOutput;

  if (clusterinput) {
    LOG(INFO) << "Insert TOF Cluster Reader";
    specs.emplace_back(o2::tof::getClusterReaderSpec(useMC));
  }
  if (trackinput) {
    LOG(INFO) << "Insert ITS-TPC Track Reader";
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
  }

  if (fitinput) {
    LOG(INFO) << "Insert FIT RecPoint Reader";
    specs.emplace_back(o2::ft0::getRecPointReaderSpec(useMC));
  }

  LOG(INFO) << "Insert TOF Matching";
  specs.emplace_back(o2::tof::getTOFRecoWorkflowSpec(useMC, useFIT));

  if (writematching && !disableRootOutput) {
    LOG(INFO) << "Insert TOF Matched Info Writer";
    specs.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC));
  }
  if (writecalib) {
    LOG(INFO) << "Insert TOF Calib Info Writer";
    specs.emplace_back(o2::tof::getTOFCalibWriterSpec());
  }

  LOG(INFO) << "Number of active devices = " << specs.size();

  return std::move(specs);
}

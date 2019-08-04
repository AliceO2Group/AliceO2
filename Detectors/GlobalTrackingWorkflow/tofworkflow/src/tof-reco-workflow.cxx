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
#include "TOFWorkflow/DigitReaderSpec.h"
#include "TOFWorkflow/ClusterReaderSpec.h"
#include "TOFWorkflow/TOFClusterizerSpec.h"
#include "TOFWorkflow/TOFClusterWriterSpec.h"
#include "TOFWorkflow/TOFMatchedWriterSpec.h"
#include "TOFWorkflow/TOFCalibWriterSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "TOFWorkflow/RecoWorkflowSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "FairLogger.h"
#include "SimConfig/ConfigurableParam.h"

// GRP
#include "DataFormatsParameters/GRPObject.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"input-type", o2::framework::VariantType::String, "digits", {"digits, raw, clusters, TBI"}});
  workflowOptions.push_back(ConfigParamSpec{"output-type", o2::framework::VariantType::String, "clusters,matching-info,calib-info", {"clusters, matching-info, calib-info, TBI"}});
  workflowOptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"disable sending of MC information, TBI"}});
  workflowOptions.push_back(ConfigParamSpec{"tof-sectors", o2::framework::VariantType::String, "0-17", {"TOF sector range, e.g. 5-7,8,9 ,TBI"}});
  workflowOptions.push_back(ConfigParamSpec{"tof-lanes", o2::framework::VariantType::Int, 1, {"number of parallel lanes up to the matcher, TBI"}});
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

  o2::base::Propagator::initFieldFromGRP("o2sim_grp.root");
  auto grp = o2::parameters::GRPObject::loadFrom("o2sim_grp.root");

  if (!grp) {
    LOG(ERROR) << "This workflow needs a valid GRP file to start";
    return specs;
  }

  //  o2::conf::ConfigurableParam::writeINI("o2tofrecoflow_configuration.ini");

  // the lane configuration defines the subspecification ids to be distributed among the lanes.
  // auto tofSectors = o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tof-sectors"));
  // std::vector<int> laneConfiguration = tofSectors;
  auto nLanes = cfgc.options().get<int>("tof-lanes");
  auto inputType = cfgc.options().get<std::string>("input-type");
  auto outputType = cfgc.options().get<std::string>("output-type");

  bool writecluster = 0;
  bool writematching = 0;
  bool writecalib = 0;

  if (outputType.rfind("clusters") < outputType.size())
    writecluster = 1;
  if (outputType.rfind("matching-info") < outputType.size())
    writematching = 1;
  if (outputType.rfind("calib-info") < outputType.size())
    writecalib = 1;

  bool clusterinput = 0;
  if (inputType == "clusters") {
    clusterinput = 1;
  }

  LOG(INFO) << "TOF RECO WORKFLOW configuration";
  LOG(INFO) << "TOF input = " << cfgc.options().get<std::string>("input-type");
  LOG(INFO) << "TOF output = " << cfgc.options().get<std::string>("output-type");
  LOG(INFO) << "TOF sectors = " << cfgc.options().get<std::string>("tof-sectors");
  LOG(INFO) << "TOF disable-mc = " << cfgc.options().get<std::string>("disable-mc");
  LOG(INFO) << "TOF lanes = " << cfgc.options().get<std::string>("tof-lanes");

  auto useMC = !cfgc.options().get<bool>("disable-mc");

  if (!clusterinput) {
    // TOF clusterizer
    LOG(INFO) << "Insert TOF Digit reader from file";
    specs.emplace_back(o2::tof::getDigitReaderSpec(useMC));
    LOG(INFO) << "Insert TOF Clusterizer";
    specs.emplace_back(o2::tof::getTOFClusterizerSpec(useMC));
    if (writecluster) {
      LOG(INFO) << "Insert TOF Cluster Writer";
      specs.emplace_back(o2::tof::getTOFClusterWriterSpec(useMC));
    }
  } else {
    LOG(INFO) << "Insert TOF Cluster Reader";
    specs.emplace_back(o2::tof::getClusterReaderSpec(useMC));
  }

  if (writematching || writecalib) {
    LOG(INFO) << "Insert ITS-TPC Track Reader";
    specs.emplace_back(o2::globaltracking::getTrackTPCITSReaderSpec(useMC));
    LOG(INFO) << "Insert TOF Matching";
    specs.emplace_back(o2::tof::getTOFRecoWorkflowSpec(useMC));

    if (writematching) {
      LOG(INFO) << "Insert TOF Matched Info Writer";
      specs.emplace_back(o2::tof::getTOFMatchedWriterSpec(useMC));
    }
    if (writecalib) {
      LOG(INFO) << "Insert TOF Calib Info Writer";
      specs.emplace_back(o2::tof::getTOFCalibWriterSpec());
    }
  }
  LOG(INFO) << "Number of active devices = " << specs.size();

  // TOF matcher
  // to be implemented

  return std::move(specs);
}

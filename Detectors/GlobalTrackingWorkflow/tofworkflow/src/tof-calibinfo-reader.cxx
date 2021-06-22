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
#include "GlobalTrackingWorkflowReaders/TrackTPCITSReaderSpec.h"
#include "TOFWorkflowIO/CalibInfoReaderSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "TOFWorkflow/RecoWorkflowSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "FairLogger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"collection-infile", o2::framework::VariantType::String, "list-calibfile", {"Name of the collection input file"}});
  workflowOptions.push_back(ConfigParamSpec{"ninstances", o2::framework::VariantType::Int, 1, {"Number of reader instances"}});
  workflowOptions.push_back(ConfigParamSpec{"tpc-matches", o2::framework::VariantType::Bool, false, {"Made from TOF-TPC matches"}});
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
    o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  }

  int ninstances = cfgc.options().get<int>("ninstances");
  auto listname = cfgc.options().get<std::string>("collection-infile");
  auto toftpc = cfgc.options().get<bool>("tpc-matches");

  char* stringTBP = new char[listname.size()];
  sprintf(stringTBP, "%s", listname.c_str());

  // the lane configuration defines the subspecification ids to be distributed among the lanes.
  // auto tofSectors = o2::RangeTokenizer::tokenize<int>(cfgc.options().get<std::string>("tof-sectors"));
  // std::vector<int> laneConfiguration = tofSectors;

  for (int i = 0; i < ninstances; i++) {
    specs.emplace_back(o2::tof::getCalibInfoReaderSpec(i, ninstances, stringTBP, toftpc));
  }

  LOG(INFO) << "Number of active devices = " << specs.size();

  return std::move(specs);
}

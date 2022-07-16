// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tof-reco-workflow.cxx
/// @author Francesco Noferini
/// @since  2019-05-22
/// @brief  Basic DPL workflow for TOF reconstruction starting from digits

#include "DetectorsBase/Propagator.h"
#include "TOFWorkflowIO/CalibInfoReaderSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "FairLogger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"collection-infile", o2::framework::VariantType::String, "list-calibfile", {"Name of the collection input file"}});
  workflowOptions.push_back(ConfigParamSpec{"ninstances", o2::framework::VariantType::Int, 1, {"Number of reader instances"}});
  workflowOptions.push_back(ConfigParamSpec{"tpc-matches", o2::framework::VariantType::Bool, false, {"Made from TOF-TPC matches"}});
  workflowOptions.push_back(ConfigParamSpec{"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings ..."}});
  o2::raw::HBFUtilsInitializer::addConfigOption(workflowOptions, o2::raw::HBFUtilsInitializer::HBFUSrc);
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

  LOG(info) << "Number of active devices = " << specs.size();
  if (ninstances == 1) {
    o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);
  } else {
    LOG(warning) << "Cannot use HBFUtilsInitializer with multiple instances";
  }
  return std::move(specs);
}

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
#include "TOFWorkflowIO/DigitReaderSpec.h"
#include "TOFWorkflowIO/TOFDigitWriterSpec.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "TOFWorkflowUtils/TOFClusterizerSpec.h"
#include "TOFWorkflowIO/TOFClusterWriterSpec.h"
#include "TOFWorkflowIO/TOFRawWriterSpec.h"
#include "TOFWorkflowUtils/CompressedDecodingTask.h"
#include "TOFWorkflowUtils/EntropyEncoderSpec.h"
#include "TOFWorkflowUtils/EntropyDecoderSpec.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "FairLogger.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"

#include <string>
#include <stdexcept>
#include <unordered_map>

void customize(std::vector<o2::framework::CallbacksPolicy>& policies)
{
  o2::raw::HBFUtilsInitializer::addNewTimeSliceCallback(policies);
}

void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TOF|tof).*[W,w]riter.*"));
}

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  std::vector<o2::framework::ConfigParamSpec> options{
    {"input-type", o2::framework::VariantType::String, "digits", {"digits, raw, clusters"}},
    {"output-type", o2::framework::VariantType::String, "clusters", {"digits, clusters, raw, ctf"}},
    {"disable-mc", o2::framework::VariantType::Bool, false, {"disable sending of MC information, TBI"}},
    {"tof-sectors", o2::framework::VariantType::String, "0-17", {"TOF sector range, e.g. 5-7,8,9 ,TBI"}},
    {"tof-lanes", o2::framework::VariantType::Int, 1, {"number of parallel lanes up to the matcher, TBI"}},
    {"use-ccdb", o2::framework::VariantType::Bool, false, {"enable access to ccdb tof calibration objects"}},
    {"input-desc", o2::framework::VariantType::String, "CRAWDATA", {"Input specs description string"}},
    {"disable-root-input", o2::framework::VariantType::Bool, false, {"disable root-files input readers"}},
    {"disable-root-output", o2::framework::VariantType::Bool, false, {"disable root-files output writers"}},
    {"conet-mode", o2::framework::VariantType::Bool, false, {"enable conet mode"}},
    {"configKeyValues", o2::framework::VariantType::String, "", {"Semicolon separated key=value strings ..."}},
    {"disable-row-writing", o2::framework::VariantType::Bool, false, {"disable ROW in Digit writing"}},
    {"write-decoding-errors", o2::framework::VariantType::Bool, false, {"trace errors in digits output when decoding"}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"calib-cluster", VariantType::Bool, false, {"to enable calib info production from clusters"}},
    {"for-calib", VariantType::Bool, false, {"to disable check on problematic, otherwise masked for new calibrations"}},
    {"cosmics", VariantType::Bool, false, {"to enable cosmics utils"}}};
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

  // the lane configuration defines the subspecification ids to be distributed among the lanes.
  auto nLanes = cfgc.options().get<int>("tof-lanes");
  auto inputType = cfgc.options().get<std::string>("input-type");
  auto outputType = cfgc.options().get<std::string>("output-type");

  bool writecluster = 0;
  bool writedigit = 0;
  bool writeraw = 0;
  bool writectf = 0;
  bool writeerr = 0;

  if (outputType.rfind("clusters") < outputType.size()) {
    writecluster = 1;
  }
  if (outputType.rfind("digits") < outputType.size()) {
    writedigit = 1;
  }
  if (outputType.rfind("raw") < outputType.size()) {
    writeraw = 1;
  }
  if (outputType.rfind("ctf") < outputType.size()) {
    writectf = 1;
  }

  bool dgtinput = 0;
  bool clusterinput = 0;
  bool rawinput = 0;
  if (inputType == "digits") {
    dgtinput = 1;
  } else if (inputType == "clusters") {
    clusterinput = 1;
  } else if (inputType == "raw") {
    rawinput = 1;
    writeerr = cfgc.options().get<bool>("write-decoding-errors");
  }

  if (!cfgc.helpOnCommandLine()) {
    o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  }

  auto useMC = !cfgc.options().get<bool>("disable-mc");
  auto useCCDB = cfgc.options().get<bool>("use-ccdb");
  bool disableRootInput = cfgc.options().get<bool>("disable-root-input") || rawinput;
  bool disableRootOutput = cfgc.options().get<bool>("disable-root-output");
  bool conetmode = cfgc.options().get<bool>("conet-mode");
  bool disableROWwriting = cfgc.options().get<bool>("disable-row-writing");
  auto isCalibFromCluster = cfgc.options().get<bool>("calib-cluster");
  auto isCosmics = cfgc.options().get<bool>("cosmics");
  auto ignoreDistStf = cfgc.options().get<bool>("ignore-dist-stf");
  auto ccdb_url = o2::base::NameConf::getCCDBServer();
  auto isForCalib = cfgc.options().get<bool>("for-calib");

  LOG(debug) << "TOF RECO WORKFLOW configuration";
  LOG(debug) << "TOF input = " << cfgc.options().get<std::string>("input-type");
  LOG(debug) << "TOF output = " << cfgc.options().get<std::string>("output-type");
  LOG(debug) << "TOF sectors = " << cfgc.options().get<std::string>("tof-sectors");
  LOG(debug) << "TOF disable-mc = " << cfgc.options().get<std::string>("disable-mc");
  LOG(debug) << "TOF lanes = " << cfgc.options().get<std::string>("tof-lanes");
  LOG(debug) << "TOF use-ccdb = " << cfgc.options().get<std::string>("use-ccdb");
  if (useCCDB) {
    LOG(debug) << "CCDB url = " << ccdb_url;
  }
  LOG(debug) << "TOF disable-root-input = " << disableRootInput;
  LOG(debug) << "TOF disable-root-output = " << disableRootOutput;
  LOG(debug) << "TOF conet-mode = " << conetmode;
  LOG(debug) << "TOF ignore Dist Stf = " << ignoreDistStf;
  LOG(debug) << "TOF disable-row-writing = " << disableROWwriting;
  LOG(debug) << "TOF write-decoding-errors = " << writeerr;

  if (clusterinput && !disableRootInput) {
    LOG(debug) << "Insert TOF Cluster Reader";
    specs.emplace_back(o2::tof::getClusterReaderSpec(useMC));
  } else if (dgtinput) {
    // TOF clusterizer
    if (!disableRootInput) {
      LOG(debug) << "Insert TOF Digit reader from file";
      specs.emplace_back(o2::tof::getDigitReaderSpec(useMC));
    }
    if (writeraw) {
      LOG(debug) << "Insert TOF Raw writer";
      specs.emplace_back(o2::tof::getTOFRawWriterSpec());
    }
  } else if (rawinput) {
    LOG(debug) << "Insert TOF Compressed Raw Decoder";
    auto inputDesc = cfgc.options().get<std::string>("input-desc");
    specs.emplace_back(o2::tof::getCompressedDecodingSpec(inputDesc, conetmode, !ignoreDistStf));
    useMC = 0;

    if (writedigit && !disableRootOutput) {
      // add TOF digit writer without mc labels
      LOG(debug) << "Insert TOF Digit Writer";
      specs.emplace_back(o2::tof::getTOFDigitWriterSpec(0, writeerr));
    }
  }

  if (!clusterinput && writecluster) {
    LOG(debug) << "Insert TOF Clusterizer";
    specs.emplace_back(o2::tof::getTOFClusterizerSpec(useMC, useCCDB, isCalibFromCluster, isCosmics, ccdb_url.c_str(), isForCalib));
    if (writecluster && !disableRootOutput) {
      LOG(debug) << "Insert TOF Cluster Writer";
      specs.emplace_back(o2::tof::getTOFClusterWriterSpec(useMC));
    }
  }

  if (writectf) {
    LOG(debug) << "Insert TOF CTF encoder";
    specs.emplace_back(o2::tof::getEntropyEncoderSpec());
  }

  LOG(debug) << "Number of active devices = " << specs.size();

  // configure dpl timer to inject correct firstTForbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, specs);

  return std::move(specs);
}

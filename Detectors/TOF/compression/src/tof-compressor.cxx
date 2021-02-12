// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   tof-compressor.cxx
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  Basic DPL workflow for TOF raw data compression

#include "TOFCompression/CompressorTask.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/Logger.h"
#include "DetectorsRaw/RDHUtils.h"

using namespace o2::framework;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  auto config = ConfigParamSpec{"tof-compressor-config", VariantType::String, "x:TOF/RAWDATA", {"TOF compressor workflow configuration"}};
  auto outputDesc = ConfigParamSpec{"tof-compressor-output-desc", VariantType::String, "CRAWDATA", {"Output specs description string"}};
  auto rdhVersion = ConfigParamSpec{"tof-compressor-rdh-version", VariantType::Int, o2::raw::RDHUtils::getVersion<o2::header::RAWDataHeader>(), {"Raw Data Header version"}};
  auto verbose = ConfigParamSpec{"tof-compressor-verbose", VariantType::Bool, false, {"Enable verbose compressor"}};

  workflowOptions.push_back(config);
  workflowOptions.push_back(outputDesc);
  workflowOptions.push_back(rdhVersion);
  workflowOptions.push_back(verbose);
}

#include "Framework/runDataProcessing.h" // the main driver

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  auto config = cfgc.options().get<std::string>("tof-compressor-config");
  //  auto outputDesc = cfgc.options().get<std::string>("output-desc");
  auto rdhVersion = cfgc.options().get<int>("tof-compressor-rdh-version");
  auto verbose = cfgc.options().get<bool>("tof-compressor-verbose");
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(OutputSpec(ConcreteDataTypeMatcher{"TOF", "CRAWDATA"}));

  AlgorithmSpec algoSpec;
  if (rdhVersion == 4) {
    if (verbose) {
      algoSpec = AlgorithmSpec{adaptFromTask<o2::tof::CompressorTask<o2::header::RAWDataHeaderV4, true>>()};
    } else {
      algoSpec = AlgorithmSpec{adaptFromTask<o2::tof::CompressorTask<o2::header::RAWDataHeaderV4, false>>()};
    }
  } else if (rdhVersion == 6) {
    if (verbose) {
      algoSpec = AlgorithmSpec{adaptFromTask<o2::tof::CompressorTask<o2::header::RAWDataHeaderV6, true>>()};
    } else {
      algoSpec = AlgorithmSpec{adaptFromTask<o2::tof::CompressorTask<o2::header::RAWDataHeaderV6, false>>()};
    }
  }

  WorkflowSpec workflow;

  /**
     We define at run time the number of devices to be attached
     to the workflow and the input matching string of the device.
     This is is done with a configuration string like the following
     one, where the input matching for each device is provide in
     comma-separated strings. For instance
     
     A:TOF/RAWDATA/768;B:TOF/RAWDATA/1024,C:TOF/RAWDATA/1280;D:TOF/RAWDATA/1536
     
     will lead to a workflow with 2 devices which will input match
     
     tof-compressor-0 --> A:TOF/RAWDATA/768;B:TOF/RAWDATA/1024
     tof-compressor-1 --> C:TOF/RAWDATA/1280;D:TOF/RAWDATA/1536
  **/

  std::stringstream ssconfig(config);
  std::string iconfig;
  int idevice = 0;

  while (getline(ssconfig, iconfig, ',')) {
    workflow.emplace_back(DataProcessorSpec{
      std::string("tof-compressor-") + std::to_string(idevice),
      select(iconfig.c_str()),
      outputs,
      algoSpec,
      Options{
        {"tof-compressor-output-buffer-size", VariantType::Int, 0, {"Encoder output buffer size (in bytes). Zero = automatic (careful)."}},
        {"tof-compressor-conet-mode", VariantType::Bool, false, {"Decoder CONET flag"}},
        {"tof-compressor-decoder-verbose", VariantType::Bool, false, {"Decoder verbose flag"}},
        {"tof-compressor-encoder-verbose", VariantType::Bool, false, {"Encoder verbose flag"}},
        {"tof-compressor-checker-verbose", VariantType::Bool, false, {"Checker verbose flag"}}}});
    idevice++;
  }

  return workflow;
}

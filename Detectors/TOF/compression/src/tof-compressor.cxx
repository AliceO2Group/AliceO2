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

#include "TOFCompression/RawReaderTask.h"
#include "TOFCompression/CompressorTask.h"
#include "TOFCompression/CompressedWriterTask.h"
#include "TOFCompression/CompressedInspectorTask.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/runDataProcessing.h" // the main driver
#include "FairLogger.h"

using namespace o2::framework;

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  //  workflowOptions.push_back(ConfigParamSpec{"verbose", o2::framework::VariantType::Bool, false, {"Verbose flag"}});
}

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  LOG(INFO) << "TOF COMPRESSION WORKFLOW configuration";

  //  auto verbose = cfgc.options().get<bool>("verbose");

  // add devices (Spec) to the workflow
  WorkflowSpec specs;
  //  specs.emplace_back(o2::tof::RawReaderTask::getSpec());
  specs.emplace_back(o2::tof::CompressorTask::getSpec());
  specs.emplace_back(o2::tof::CompressedWriterTask::getSpec());
  specs.emplace_back(o2::tof::CompressedInspectorTask::getSpec());

  LOG(INFO) << "Number of active devices = " << specs.size();
  return std::move(specs);
}

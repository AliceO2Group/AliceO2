// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   crucompressor.cxx
/// @author Sean Murray
/// @brief  Basic DPL workflow for TRD CRU output(raw) to tracklet/digit data.
///         There may or may not be some compression in this at some point.

#include "TRDReconstruction/CruCompressorTask.h"
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
  auto config = ConfigParamSpec{"trd-crucompressor-config", VariantType::String, "A:TRD/RAWDATA", {"TRD raw data config"}};
  auto outputDesc = ConfigParamSpec{"trd-crucompressor-output-desc", VariantType::String, "TRDTLT", {"Output specs description string"}};
  auto verbosity = ConfigParamSpec{"trd-crucompressor-verbose", VariantType::Bool, false, {"Enable verbose compressor"}};
  auto verboseheaders = ConfigParamSpec{"trd-crucompressor-hedaerverbose", VariantType::Bool, false, {"Enable header verbose compressor"}};
  auto verbosedata = ConfigParamSpec{"trd-crucompressor-dataverbose", VariantType::Bool, false, {"Enable data verbose compressor"}};

  auto digithcheader = ConfigParamSpec{"trd-crucompressor-digitheader", VariantType::Bool, true, {"using digit half chamber headers"}};
  auto tracklethcheader = ConfigParamSpec{"trd-crucompressor-tracklethcheader", VariantType::Bool, true, {"using tracklet half chamber headers"}};
  auto trackletformat = ConfigParamSpec{"trd-crucompressor-trackletformat", VariantType::Int, 0, {"0: no pid scale factor, 1: pid scale factor"}};
  auto digitformat = ConfigParamSpec{"trd-crucompressor-digitformat", VariantType::Int, 1, {"0: zero supressed digits, 1: non zero suppressed digits "}};

  workflowOptions.push_back(config);
  workflowOptions.push_back(outputDesc);
  workflowOptions.push_back(verbosity);
  workflowOptions.push_back(verboseheaders);
  workflowOptions.push_back(verbosedata);
  workflowOptions.push_back(digithcheader);
  workflowOptions.push_back(tracklethcheader);
  workflowOptions.push_back(trackletformat);
  workflowOptions.push_back(digitformat);
}

#include "Framework/runDataProcessing.h" // the main driver

/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  auto config = cfgc.options().get<std::string>("trd-crucompressor-config");
  auto verbosity = cfgc.options().get<bool>("trd-crucompressor-verbose");
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(OutputSpec(ConcreteDataTypeMatcher{"TRD", "CDATA"}));

  AlgorithmSpec algoSpec;
  algoSpec = AlgorithmSpec{adaptFromTask<o2::trd::CruCompressorTask>()};

  WorkflowSpec workflow;

  /*
   * This is originaly replicated from TOF
     We define at run time the number of devices to be attached
     to the workflow and the input matching string of the device.
     This is is done with a configuration string like the following
     one, where the input matching for each device is provide in
     comma-separated strings. For instance
     A:TRD/RAWDATA/785;B:TRF/RAWDATA/2560,C:TRD/RAWDATA/1280;D:TRD/RAWDATA/1536

     will lead to a workflow with 2 devices which will input match

     trd-crucompressor-0 --> A:TRD/RAWDATA/768;B:TRD/RAWDATA/1024
     trd-crucompressor-1 --> C:TRD/RAWDATA/1280;D:TRD/RAWDATA/1536
     The number after the RAWDATA is the FeeID in decimal
  */

  std::stringstream ssconfig(config);
  std::string iconfig;
  int idevice = 0;
  LOG(info) << " config string is : " << config;
  LOG(info) << "for now ignoring the multiple processors, something going wrong";
  while (getline(ssconfig, iconfig, ',')) { // for now we will keep the possibilty to have a device per half cru/feeid i.e. 6 per flp
                                            // this is probably never going to be used but would to nice to know hence here.
    workflow.emplace_back(DataProcessorSpec{
      std::string("trd-crucompressor-") + std::to_string(idevice),
      select(iconfig.c_str()),
      outputs,
      algoSpec,
      Options{
        {"trd-crucompressor-verbose", VariantType::Bool, false, {"verbose flag"}}}});
    idevice++;
  }

  return workflow;
}

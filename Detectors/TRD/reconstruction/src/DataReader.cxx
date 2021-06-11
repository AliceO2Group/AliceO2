// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   datareader.cxx
/// @author Sean Murray
/// @brief  Basic DPL workflow for TRD CRU output(raw) or compressed format to tracklet data.
///         There may or may not be some compression in this at some point.

#include "TRDReconstruction/DataReaderTask.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConcreteDataMatcher.h"
#include "DetectorsRaw/HBFUtilsInitializer.h"
#include "Framework/Logger.h"
#include "DetectorsRaw/RDHUtils.h"

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{

  std::vector<o2::framework::ConfigParamSpec> options{
    {"trd-datareader-inputspec", VariantType::String, "RAWDATA", {"TRD raw data spec"}},
    {"trd-datareader-output-desc", VariantType::String, "TRDTLT", {"Output specs description string"}},
    {"trd-datareader-verbose", VariantType::Bool, false, {"Enable verbose epn data reading"}},
    {"trd-datareader-headerverbose", VariantType::Bool, false, {"Enable verbose header info"}},
    {"trd-datareader-dataverbose", VariantType::Bool, false, {"Enable verbose data info"}},
    {"trd-datareader-compresseddata", VariantType::Bool, false, {"The incoming data is compressed or not"}},
    {"trd-datareader-enablebyteswapdata", VariantType::Bool, false, {"byteswap the incoming data, raw data needs it and simulation does not."}}};

  o2::raw::HBFUtilsInitializer::addConfigOption(options);

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;
/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  //  auto config = cfgc.options().get<std::string>("trd-datareader-config");
  //
  //
  //  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  // o2::conf::ConfigurableParam::writeINI("o2trdrawreader-workflow_configuration.ini");

  auto inputspec = cfgc.options().get<std::string>("trd-datareader-inputspec");
  //auto outputspec = cfgc.options().get<std::string>("trd-datareader-outputspec");
  auto verbose = cfgc.options().get<bool>("trd-datareader-verbose");
  auto byteswap = cfgc.options().get<bool>("trd-datareader-enablebyteswapdata");
  auto compresseddata = cfgc.options().get<bool>("trd-datareader-compresseddata");
  auto headerverbose = cfgc.options().get<bool>("trd-datareader-headerverbose");
  auto dataverbose = cfgc.options().get<bool>("trd-datareader-dataverbose");

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "TRACKLETS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  //outputs.emplace_back("TRD", "FLPSTAT", 0, Lifetime::Timeframe);
  LOG(info) << "input spec is:" << inputspec;
  LOG(info) << "enablebyteswap :" << byteswap;
  AlgorithmSpec algoSpec;
  algoSpec = AlgorithmSpec{adaptFromTask<o2::trd::DataReaderTask>(compresseddata, byteswap, verbose, headerverbose, dataverbose)};

  WorkflowSpec workflow;

  /*
   * This is originally replicated from TOF
     We define at run time the number of devices to be attached
     to the workflow and the input matching string of the device.
     This is is done with a configuration string like the following
     one, where the input matching for each device is provide in
     comma-separated strings. For instance
  */

  //  std::stringstream ssconfig(inputspec);
  std::string iconfig;
  std::string inputDescription;
  int idevice = 0;
  //  LOG(info) << "expected incoming data definition : " << inputspec;
  // this is probably never going to be used but would to nice to know hence here.
  workflow.emplace_back(DataProcessorSpec{
    std::string("trd-datareader"), // left as a string cast incase we append stuff to the string
    select(std::string("x:TRD/" + inputspec).c_str()),
    outputs,
    algoSpec,
    Options{}});

  // configure dpl timer to inject correct firstTFOrbit: start from the 1st orbit of TF containing 1st sampled orbit
  o2::raw::HBFUtilsInitializer hbfIni(cfgc, workflow);

  return workflow;
}

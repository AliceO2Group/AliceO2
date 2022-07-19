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

/// @file   datareader.cxx
/// @author Sean Murray
/// @brief  Basic DPL workflow for TRD CRU output(raw) or compressed format to tracklet data.
///         There may or may not be some compression in this at some point.

#include "TRDReconstruction/DataReaderTask.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConcreteDataMatcher.h"
#include "Framework/CallbacksPolicy.h"
#include "Framework/Logger.h"
#include "Framework/CCDBParamSpec.h"
#include "DetectorsRaw/RDHUtils.h"
#include "TRDWorkflowIO/TRDTrackletWriterSpec.h"
#include "TRDWorkflowIO/TRDDigitWriterSpec.h"
#include "DataFormatsTRD/RawDataStats.h"

// add workflow options, note that customization needs to be declared before
// including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{

  std::vector<o2::framework::ConfigParamSpec> options{
    {"output-desc", VariantType::String, "TRDTLT", {"Output specs description string."}},
    {"verbose", VariantType::Bool, false, {"Enable verbose epn data reading."}},
    {"verbosehalfcru", VariantType::Bool, false, {"Enable verbose for a halfcru, the halfcru contents are dumped out in hex."}},
    {"verboselink", VariantType::Bool, false, {"Enable verbose for a link, the links contents are dumped out in hex."}},
    {"verboseword", VariantType::Bool, false, {"Enable verbose for each word seen, as its seen, labeled, identified/rejected, and unpacked."}},
    {"verboseerrors", VariantType::Bool, false, {"Enable verbose error text, instead of simply updating the spectra."}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"fixdigitcorruptdata", VariantType::Bool, false, {"Fix the erroneous data at the end of digits"}},
    {"ignore-tracklethcheader", VariantType::Bool, false, {"Ignore the tracklethalf chamber header for cross referencing"}},
    {"halfchamberwords", VariantType::Int, 0, {"Fix half chamber for when it is version is 0.0 integer value of additional header words, ignored if version is not 0.0"}},
    {"halfchambermajor", VariantType::Int, 0, {"Fix half chamber for when it is version is 0.0 integer value of major version, ignored if version is not 0.0"}},
    {"ignore-digithcheader", VariantType::Bool, false, {"Ignore the digithalf chamber header for cross referencing, take rdh/cru as authorative."}},
    {"fixforoldtrigger", VariantType::Bool, false, {"Fix for the old data not having a 2 stage trigger stored in the cru header."}},
    {"onlycalibrationtrigger", VariantType::Bool, false, {"Only permit calibration triggers, used for debugging traclets and their digits, maybe other uses."}},
    {"tracklethcheader", VariantType::Int, 2, {"Status of TrackletHalfChamberHeader 0 off always, 1 iff tracklet data, 2 on always"}},
    {"generate-stats", VariantType::Bool, true, {"Generate the state message sent to qc"}},
    {"enablebyteswapdata", VariantType::Bool, false, {"byteswap the incoming data, raw data needs it and simulation does not."}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;
/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  //  auto config = cfgc.options().get<std::string>("trd-datareader-config");
  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));
  //auto outputspec = cfgc.options().get<std::string>("trd-datareader-outputspec");
  auto askSTFDist = !cfgc.options().get<bool>("ignore-dist-stf");
  auto tracklethcheader = cfgc.options().get<int>("tracklethcheader");
  auto halfchamberwords = cfgc.options().get<int>("halfchamberwords");
  auto halfchambermajor = cfgc.options().get<int>("halfchambermajor");

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "TRACKLETS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "RAWSTATS", 0, Lifetime::Timeframe);
  //outputs.emplace_back("TRD", "FLPSTAT", 0, Lifetime::Timeframe);
  //
  std::bitset<16> binaryoptions;
  binaryoptions[o2::trd::TRDVerboseBit] = cfgc.options().get<bool>("verbose");
  binaryoptions[o2::trd::TRDVerboseLinkBit] = cfgc.options().get<bool>("verboselink");
  binaryoptions[o2::trd::TRDVerboseHalfCruBit] = cfgc.options().get<bool>("verbosehalfcru");
  binaryoptions[o2::trd::TRDVerboseWordBit] = cfgc.options().get<bool>("verboseword");
  binaryoptions[o2::trd::TRDVerboseErrorsBit] = cfgc.options().get<bool>("verboseerrors");
  binaryoptions[o2::trd::TRDFixDigitCorruptionBit] = cfgc.options().get<bool>("fixdigitcorruptdata");
  binaryoptions[o2::trd::TRDIgnoreDigitHCHeaderBit] = cfgc.options().get<bool>("ignore-digithcheader");
  binaryoptions[o2::trd::TRDIgnoreTrackletHCHeaderBit] = cfgc.options().get<bool>("ignore-tracklethcheader");
  binaryoptions[o2::trd::TRDByteSwapBit] = cfgc.options().get<bool>("enablebyteswapdata");
  binaryoptions[o2::trd::TRDIgnore2StageTrigger] = cfgc.options().get<bool>("fixforoldtrigger");
  binaryoptions[o2::trd::TRDGenerateStats] = cfgc.options().get<bool>("generate-stats");
  binaryoptions[o2::trd::TRDOnlyCalibrationTriggerBit] = cfgc.options().get<bool>("onlycalibrationtrigger");
  AlgorithmSpec algoSpec;
  algoSpec = AlgorithmSpec{adaptFromTask<o2::trd::DataReaderTask>(tracklethcheader, halfchamberwords, halfchambermajor, binaryoptions)};

  WorkflowSpec workflow;

  std::string iconfig;
  std::string inputDescription;
  int idevice = 0;
  auto orig = o2::header::gDataOriginTRD;
  auto inputs = o2::framework::select(std::string("x:TRD/RAWDATA").c_str());
  for (auto& inp : inputs) {
    // take care of case where our data is not in the time frame
    inp.lifetime = Lifetime::Optional;
  }
  if (askSTFDist) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CTP/Config/TriggerOffsets"));
  workflow.emplace_back(DataProcessorSpec{
    std::string("trd-datareader"), // left as a string cast incase we append stuff to the string
    inputs,
    outputs,
    algoSpec,
    Options{{"log-max-errors", VariantType::Int, 20, {"maximum number of errors to log"}},
            {"log-max-warnings", VariantType::Int, 20, {"maximum number of warnings to log"}}}});

  return workflow;
}

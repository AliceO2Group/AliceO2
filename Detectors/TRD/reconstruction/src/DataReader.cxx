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
    {"trd-datareader-output-desc", VariantType::String, "TRDTLT", {"Output specs description string"}},
    {"trd-datareader-verbose", VariantType::Bool, false, {"Enable verbose epn data reading"}},
    {"trd-datareader-headerverbose", VariantType::Bool, false, {"Enable verbose header info"}},
    {"trd-datareader-dataverbose", VariantType::Bool, false, {"Enable verbose data info"}},
    {"trd-datareader-compresseddata", VariantType::Bool, false, {"The incoming data is compressed or not"}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"trd-datareader-fixdigitcorruptdata", VariantType::Bool, false, {"Fix the erroneous data at the end of digits"}},
    {"enable-timing", VariantType::Bool, false, {"enable the timing of tracklet, digit, timeframe, cru processing"}},
    {"enable-stats", VariantType::Bool, false, {"enable the reader stats"}},
    {"enable-root-output", VariantType::Bool, false, {"Write the data to file"}},
    {"ignore-tracklethcheader", VariantType::Bool, false, {"Ignore the tracklethalf chamber header for cross referencing"}},
    {"halfchamberwords", VariantType::Int, 0, {"Fix half chamber for when it is version is 0.0 integer value of additional header words, ignored if version is not 0.0"}},
    {"halfchambermajor", VariantType::Int, 0, {"Fix half chamber for when it is version is 0.0 integer value of major version, ignored if version is not 0.0"}},
    {"ignore-digithcheader", VariantType::Bool, false, {"Ignore the digithalf chamber header for cross referencing, take rdh/cru as authorative."}},
    {"fixsm1617", VariantType::Bool, false, {"Fix the missing bit in the tracklet hc header of supermodules 16 and 17. Requires option tracklethcheader 2"}},
    {"fixforoldtrigger", VariantType::Bool, false, {"Fix for the old data not having a 2 stage trigger stored in the cru header."}},
    {"tracklethcheader", VariantType::Int, 2, {"Status of TrackletHalfChamberHeader 0 off always, 1 iff tracklet data, 2 on always"}},
    {"histogramsfile", VariantType::String, "histos.root", {"Name of the histogram file, so one can run multiple per node"}},
    {"trd-debugm1", VariantType::Bool, false, {"various output statements specific to debugging m1 problems."}},
    //{"generate-stats", VariantType::Bool, true, {"Generate the state message sent to qc"}},
    {"trd-datareader-enablebyteswapdata", VariantType::Bool, false, {"byteswap the incoming data, raw data needs it and simulation does not."}},
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
  auto verbose = cfgc.options().get<bool>("trd-datareader-verbose");
  auto byteswap = cfgc.options().get<bool>("trd-datareader-enablebyteswapdata");
  auto compresseddata = cfgc.options().get<bool>("trd-datareader-compresseddata");
  auto headerverbose = cfgc.options().get<bool>("trd-datareader-headerverbose");
  auto dataverbose = cfgc.options().get<bool>("trd-datareader-dataverbose");
  auto askSTFDist = !cfgc.options().get<bool>("ignore-dist-stf");
  auto fixdigitcorruption = cfgc.options().get<bool>("trd-datareader-fixdigitcorruptdata");
  auto tracklethcheader = cfgc.options().get<int>("tracklethcheader");
  auto enabletimeinfo = cfgc.options().get<bool>("enable-timing");
  auto enablestats = cfgc.options().get<bool>("enable-stats");
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
  binaryoptions[o2::trd::TRDVerboseBit] = cfgc.options().get<bool>("trd-datareader-verbose");
  binaryoptions[o2::trd::TRDHeaderVerboseBit] = cfgc.options().get<bool>("trd-datareader-headerverbose");
  binaryoptions[o2::trd::TRDDataVerboseBit] = cfgc.options().get<bool>("trd-datareader-dataverbose");
  binaryoptions[o2::trd::TRDCompressedDataBit] = cfgc.options().get<bool>("trd-datareader-compresseddata");
  binaryoptions[o2::trd::TRDFixDigitCorruptionBit] = cfgc.options().get<bool>("trd-datareader-fixdigitcorruptdata");
  binaryoptions[o2::trd::TRDEnableTimeInfoBit] = cfgc.options().get<bool>("enable-timing");
  binaryoptions[o2::trd::TRDEnableStatsBit] = cfgc.options().get<bool>("enable-stats");
  binaryoptions[o2::trd::TRDIgnoreDigitHCHeaderBit] = cfgc.options().get<bool>("ignore-digithcheader");
  binaryoptions[o2::trd::TRDIgnoreTrackletHCHeaderBit] = cfgc.options().get<bool>("ignore-tracklethcheader");
  binaryoptions[o2::trd::TRDEnableRootOutputBit] = cfgc.options().get<bool>("enable-root-output");
  binaryoptions[o2::trd::TRDByteSwapBit] = cfgc.options().get<bool>("trd-datareader-enablebyteswapdata");
  //binaryoptions[o2::trd::TRDFixSM1617Bit] = cfgc.options().get<bool>("fixsm1617");
  binaryoptions[o2::trd::TRDIgnore2StageTrigger] = cfgc.options().get<bool>("fixforoldtrigger");
  //binaryoptions[o2::trd::TRDGenerateStats] = cfgc.options().get<bool>("generate-stats");
  binaryoptions[o2::trd::TRDGenerateStats] = true; //cfgc.options().get<bool>("generate-stats");
  binaryoptions[o2::trd::TRDM1Debug] = cfgc.options().get<bool>("trd-debugm1");
  AlgorithmSpec algoSpec;
  algoSpec = AlgorithmSpec{adaptFromTask<o2::trd::DataReaderTask>(tracklethcheader, halfchamberwords, halfchambermajor, cfgc.options().get<std::string>("histogramsfile"), binaryoptions)};

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

  if (cfgc.options().get<bool>("enable-root-output")) {
    workflow.emplace_back(o2::trd::getTRDDigitWriterSpec(false, false));
    workflow.emplace_back(o2::trd::getTRDTrackletWriterSpec(false));
  }

  return workflow;
}

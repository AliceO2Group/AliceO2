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
    {"verbose", VariantType::Bool, false, {"Enable verbose epn data reading."}},
    {"verboseerrors", VariantType::Bool, false, {"Enable verbose error text, instead of simply updating the spectra."}},
    {"ignore-dist-stf", VariantType::Bool, false, {"do not subscribe to FLP/DISTSUBTIMEFRAME/0 message (no lost TF recovery)"}},
    {"halfchamberwords", VariantType::Int, 0, {"Fix half chamber for when it is version is 0.0 integer value of additional header words, ignored if version is not 0.0"}},
    {"halfchambermajor", VariantType::Int, 0, {"Fix half chamber for when it is version is 0.0 integer value of major version, ignored if version is not 0.0"}},
    {"fixforoldtrigger", VariantType::Bool, false, {"Fix for the old data not having a 2 stage trigger stored in the cru header."}},
    {"onlycalibrationtrigger", VariantType::Bool, false, {"Only permit calibration triggers, used for debugging traclets and their digits, maybe other uses."}},
    {"tracklethcheader", VariantType::Int, 2, {"Status of TrackletHalfChamberHeader 0 off always, 1 iff tracklet data, 2 on always"}},
    {"generate-stats", VariantType::Bool, true, {"Generate the state message sent to qc"}},
    {"disable-root-output", VariantType::Bool, false, {"Do not write the digits and tracklets to file"}},
    {"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}}};
  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h" // the main driver

using namespace o2::framework;
/// This function hooks up the the workflow specifications into the DPL driver.
WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{

  o2::conf::ConfigurableParam::updateFromString(cfgc.options().get<std::string>("configKeyValues"));

  auto askSTFDist = !cfgc.options().get<bool>("ignore-dist-stf");
  auto tracklethcheader = cfgc.options().get<int>("tracklethcheader");
  auto halfchamberwords = cfgc.options().get<int>("halfchamberwords");
  auto halfchambermajor = cfgc.options().get<int>("halfchambermajor");

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRD", "TRACKLETS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "DIGITS", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  outputs.emplace_back("TRD", "RAWSTATS", 0, Lifetime::Timeframe);

  std::bitset<16> binaryoptions;
  binaryoptions[o2::trd::TRDVerboseBit] = cfgc.options().get<bool>("verbose");
  binaryoptions[o2::trd::TRDVerboseErrorsBit] = cfgc.options().get<bool>("verboseerrors");
  binaryoptions[o2::trd::TRDIgnore2StageTrigger] = cfgc.options().get<bool>("fixforoldtrigger");
  binaryoptions[o2::trd::TRDGenerateStats] = cfgc.options().get<bool>("generate-stats");
  binaryoptions[o2::trd::TRDOnlyCalibrationTriggerBit] = cfgc.options().get<bool>("onlycalibrationtrigger");
  binaryoptions[o2::trd::TRDDisableRootOutputBit] = cfgc.options().get<bool>("disable-root-output");
  AlgorithmSpec algoSpec;
  algoSpec = AlgorithmSpec{adaptFromTask<o2::trd::DataReaderTask>(tracklethcheader, halfchamberwords, halfchambermajor, binaryoptions)};

  WorkflowSpec workflow;

  auto inputs = o2::framework::select(std::string("x:TRD/RAWDATA").c_str());
  for (auto& inp : inputs) {
    // take care of case where our data is not in the time frame
    inp.lifetime = Lifetime::Optional;
  }
  if (askSTFDist) {
    inputs.emplace_back("stdDist", "FLP", "DISTSUBTIMEFRAME", 0, Lifetime::Timeframe);
  }
  inputs.emplace_back("trigoffset", "CTP", "Trig_Offset", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CTP/Config/TriggerOffsets"));
  inputs.emplace_back("linkToHcid", "TRD", "LinkToHcid", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("TRD/Config/LinkToHCIDMapping"));
  workflow.emplace_back(DataProcessorSpec{
    std::string("trd-datareader"), // left as a string cast incase we append stuff to the string
    inputs,
    outputs,
    algoSpec,
    Options{{"log-max-errors", VariantType::Int, 20, {"maximum number of errors to log"}},
            {"log-max-warnings", VariantType::Int, 20, {"maximum number of warnings to log"}}}});

  if (!cfgc.options().get<bool>("disable-root-output")) {
    workflow.emplace_back(o2::trd::getTRDDigitWriterSpec(false, false));
    workflow.emplace_back(o2::trd::getTRDTrackletWriterSpec(false));
  }

  return workflow;
}

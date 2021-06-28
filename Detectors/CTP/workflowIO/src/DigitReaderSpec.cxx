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

#include "CTPWorkflowIO/DigitReaderSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Headers/DataHeader.h"
#include "DPLUtils/RootTreeReader.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/DataSpecUtils.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include <memory>
#include <utility>

using namespace o2::framework;

namespace o2
{

namespace ctp
{

struct ProcessAttributes {
  std::shared_ptr<RootTreeReader> reader;
  std::string datatype;
  bool terminateOnEod;
  bool finished;
};

DataProcessorSpec getDigitsReaderSpec(bool propagateMC)
{
  if (propagateMC) {
    LOG(WARNING) << "MC truth not implemented for CTP, continouing wothout MC";
    propagateMC = false;
  }
  auto initFunction = [propagateMC](InitContext& ic) {
    // get the option from the init context
    auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                  ic.options().get<std::string>("infile"));
    auto treename = ic.options().get<std::string>("treename");
    auto nofEvents = ic.options().get<int>("nevents");
    auto publishingMode = nofEvents == -1 ? RootTreeReader::PublishingMode::Single : RootTreeReader::PublishingMode::Loop;
    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      processAttributes->terminateOnEod = ic.options().get<bool>("terminate-on-eod");
      processAttributes->finished = false;
      processAttributes->datatype = "CTPDigit";
      constexpr auto persistency = Lifetime::Timeframe;
      o2::header::DataHeader::SubSpecificationType subSpec = 0;
      if (propagateMC) {
        //processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
        //filename.c_str(), // input file name
        //nofEvents,        // number of entries to publish
        //publishingMode,
        //Output{"CTP", "DIGITS", subSpec, persistency},
        //"CTPDigit", // name of data branch
        //Output{"CTP", "DIGITSMCTR", subSpec, persistency},"CPVDigitMCTruth");
      }
      processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                   filename.c_str(), // input file name
                                                                   nofEvents,        // number of entries to publish
                                                                   publishingMode,
                                                                   Output{"CTP", "DIGITS", subSpec, persistency},
                                                                   "CTPDigit"); // name of data branch
    }

    auto processFunction = [processAttributes, propagateMC](ProcessingContext& pc) { // false for propagateMC
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &pc, propagateMC]() { // false for propgateMC
        //o2::cpv::CPVBlockHeader cpvheader(true);
        //if (processAttributes->reader->next()) {
        //(*processAttributes->reader)(pc, cpvheader);
        //} else {
        //processAttributes->reader.reset();
        //return false;
        //}
        return true;
      };

      bool active(true);
      if (!publish()) {
        active = false;
        // Send dummy header with no payload option
        //o2::cpv::CPVBlockHeader dummyheader(false);
        //pc.outputs().snapshot(OutputRef{"output", 0, {dummyheader}}, 0);
        //pc.outputs().snapshot(OutputRef{"outputTR", 0, {dummyheader}}, 0);
        //if (propagateMC) {
        //pc.outputs().snapshot(OutputRef{"outputMC", 0, {dummyheader}}, 0);
        //}
      }
      if ((processAttributes->finished = (active == false)) && processAttributes->terminateOnEod) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(framework::QuitRequest::Me);
      }
    };
    return processFunction;
  };

  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(OutputSpec{{"output"}, "CTP", "DIGITS", 0, Lifetime::Timeframe});
  if (propagateMC) {
    outputSpecs.emplace_back(OutputSpec{{"outputMC"}, "CTP", "DIGITSMCTR", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "ctp-digit-reader",
    Inputs{}, // no inputs
    outputSpecs,
    AlgorithmSpec(initFunction),
    Options{
      {"infile", VariantType::String, "ctpdigits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "o2sim", {"Name of input tree"}},
      {"nevents", VariantType::Int, -1, {"number of events to run, -1: inf loop"}},
      {"terminate-on-eod", VariantType::Bool, true, {"terminate on end-of-data"}},
    }};
}
} // namespace ctp

} // namespace o2

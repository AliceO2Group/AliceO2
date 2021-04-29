// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsPHOS/PHOSBlockHeader.h"
#include "PHOSWorkflow/ReaderSpec.h"
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

namespace phos
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

struct ProcessAttributes {
  std::shared_ptr<RootTreeReader> reader;
  std::string datatype;
  bool terminateOnEod;
  bool finished;
};

DataProcessorSpec getDigitsReaderSpec(bool propagateMC)
{

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
      processAttributes->datatype = "PHOSDigit";
      constexpr auto persistency = Lifetime::Timeframe;
      o2::header::DataHeader::SubSpecificationType subSpec = 0;
      if (propagateMC) {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     Output{"PHS", "DIGITS", subSpec, persistency},
                                                                     "PHOSDigit", // name of data branch
                                                                     Output{"PHS", "DIGITTRIGREC", subSpec, persistency},
                                                                     "PHOSDigitTrigRecords", // name of data triggerrecords branch
                                                                     Output{"PHS", "DIGITSMCTR", subSpec, persistency},
                                                                     "PHOSDigitMCTruth"); // name of mc label branch
      } else {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     Output{"PHS", "DIGITS", subSpec, persistency},
                                                                     "PHOSDigit", // name of data branch
                                                                     Output{"PHS", "DIGITTRIGREC", subSpec, persistency},
                                                                     "PHOSDigitTrigRecords"); // name of data triggerrecords branch
      }
    }

    auto processFunction = [processAttributes, propagateMC](ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &pc, propagateMC]() {
        o2::phos::PHOSBlockHeader phosheader(true);
        if (processAttributes->reader->next()) {
          (*processAttributes->reader)(pc, phosheader);
        } else {
          processAttributes->reader.reset();
          return false;
        }
        return true;
      };

      bool active(true);
      if (!publish()) {
        active = false;
        // Send dummy header with no payload option
        o2::phos::PHOSBlockHeader dummyheader(false);
        pc.outputs().snapshot(OutputRef{"output", 0, {dummyheader}}, 0);
        pc.outputs().snapshot(OutputRef{"outputTR", 0, {dummyheader}}, 0);
        if (propagateMC) {
          pc.outputs().snapshot(OutputRef{"outputMC", 0, {dummyheader}}, 0);
        }
      }
      if ((processAttributes->finished = (active == false)) && processAttributes->terminateOnEod) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(framework::QuitRequest::Me);
      }
    };
    return processFunction;
  };

  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(OutputSpec{{"output"}, "PHS", "DIGITS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"outputTR"}, "PHS", "DIGITTRIGREC", 0, Lifetime::Timeframe});
  if (propagateMC) {
    outputSpecs.emplace_back(OutputSpec{{"outputMC"}, "PHS", "DIGITSMCTR", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "phos-digit-reader",
    Inputs{}, // no inputs
    outputSpecs,
    AlgorithmSpec(initFunction),
    Options{
      {"infile", VariantType::String, "phosdigits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "o2sim", {"Name of input tree"}},
      {"nevents", VariantType::Int, -1, {"number of events to run, -1: inf loop"}},
      {"terminate-on-eod", VariantType::Bool, true, {"terminate on end-of-data"}},
    }};
}

///////////////Cell reader

DataProcessorSpec getCellReaderSpec(bool propagateMC)
{

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
      processAttributes->datatype = "PHOSCell";
      constexpr auto persistency = Lifetime::Timeframe;
      o2::header::DataHeader::SubSpecificationType subSpec = 0;
      if (propagateMC) {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     Output{"PHS", "CELLS", subSpec, persistency},
                                                                     "PHOSCell", // name of data branch
                                                                     Output{"PHS", "CELLTRIGREC", subSpec, persistency},
                                                                     "PHOSCellTrigRec", // name of data triggerrecords branch
                                                                     Output{"PHS", "CELLSMCTR", subSpec, persistency},
                                                                     "PHOSCellTrueMC"); // name of mc label branch
      } else {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     Output{"PHS", "CELLS", subSpec, persistency},
                                                                     "PHOSCell", // name of data branch
                                                                     Output{"PHS", "CELLTRIGREC", subSpec, persistency},
                                                                     "PHOSCellTrigRec"); // name of data triggerrecords branch
      }
    }

    auto processFunction = [processAttributes, propagateMC](ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &pc, propagateMC]() {
        PHOSBlockHeader phosheader(true);
        if (processAttributes->reader->next()) {
          (*processAttributes->reader)(pc, phosheader);
        } else {
          processAttributes->reader.reset();
          return false;
        }
        return true;
      };

      bool active(true);
      if (!publish()) {
        active = false;
        // Send dummy header with no payload option
        PHOSBlockHeader dummyheader(false);
        pc.outputs().snapshot(OutputRef{"output", 0, {dummyheader}}, 0);
        pc.outputs().snapshot(OutputRef{"outputTR", 0, {dummyheader}}, 0);
        if (propagateMC) {
          pc.outputs().snapshot(OutputRef{"outputMC", 0, {dummyheader}}, 0);
          pc.outputs().snapshot(OutputRef{"outputMCmap", 0, {dummyheader}}, 0);
        }
      }
      if ((processAttributes->finished = (active == false)) && processAttributes->terminateOnEod) {
        pc.services().get<ControlService>().endOfStream();
        pc.services().get<ControlService>().readyToQuit(framework::QuitRequest::Me);
      }
    };
    return processFunction;
  };

  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(OutputSpec{{"output"}, "PHS", "CELLS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"outputTR"}, "PHS", "CELLTRIGREC", 0, Lifetime::Timeframe});
  if (propagateMC) {
    outputSpecs.emplace_back(OutputSpec{{"outputMC"}, "PHS", "CELLSMCTR", 0, Lifetime::Timeframe});
    outputSpecs.emplace_back(OutputSpec{{"outputMCmap"}, "PHS", "CELLSMCMAP", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "phos-cell-reader",
    Inputs{}, // no inputs
    outputSpecs,
    AlgorithmSpec(initFunction),
    Options{
      {"infile", VariantType::String, "phoscells.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "o2sim", {"Name of input tree"}},
      {"nevents", VariantType::Int, -1, {"number of events to run, -1: inf loop"}},
      {"terminate-on-eod", VariantType::Bool, true, {"terminate on end-of-data"}},
    }};
}

} // namespace phos

} // namespace o2

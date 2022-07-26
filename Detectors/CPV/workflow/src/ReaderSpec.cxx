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

#include "DataFormatsCPV/CPVBlockHeader.h"
#include "DataFormatsCPV/Cluster.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "CPVWorkflow/ReaderSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Headers/DataHeader.h"
#include "DPLUtils/RootTreeReader.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/DataSpecUtils.h"
#include "CommonUtils/NameConf.h"
#include <memory>
#include <utility>

using namespace o2::framework;

namespace o2
{

namespace cpv
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
      processAttributes->datatype = "CPVDigit";
      o2::header::DataHeader::SubSpecificationType subSpec = 0;
      if (propagateMC) {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"CPV", "DIGITS", subSpec}, "CPVDigit"},
                                                                     RootTreeReader::BranchDefinition<std::vector<TriggerRecord>>{Output{"CPV", "DIGITTRIGREC", subSpec}, "CPVDigitTrigRecords"},
                                                                     RootTreeReader::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{Output{"CPV", "DIGITSMCTR", subSpec}, "CPVDigitMCTruth"});
      } else {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     RootTreeReader::BranchDefinition<std::vector<Digit>>{Output{"CPV", "DIGITS", subSpec}, "CPVDigit"},
                                                                     RootTreeReader::BranchDefinition<std::vector<TriggerRecord>>{Output{"CPV", "DIGITTRIGREC", subSpec}, "CPVDigitTrigRecords"});
      }
    }

    auto processFunction = [processAttributes, propagateMC](ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &pc, propagateMC]() {
        o2::cpv::CPVBlockHeader cpvheader(true);
        if (processAttributes->reader->next()) {
          (*processAttributes->reader)(pc, cpvheader);
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
        o2::cpv::CPVBlockHeader dummyheader(false);
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
  outputSpecs.emplace_back(OutputSpec{{"output"}, "CPV", "DIGITS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"outputTR"}, "CPV", "DIGITTRIGREC", 0, Lifetime::Timeframe});
  if (propagateMC) {
    outputSpecs.emplace_back(OutputSpec{{"outputMC"}, "CPV", "DIGITSMCTR", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "cpv-digit-reader",
    Inputs{}, // no inputs
    outputSpecs,
    AlgorithmSpec(initFunction),
    Options{
      {"infile", VariantType::String, "cpvdigits.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "o2sim", {"Name of input tree"}},
      {"nevents", VariantType::Int, -1, {"number of events to run, -1: inf loop"}},
      {"terminate-on-eod", VariantType::Bool, true, {"terminate on end-of-data"}},
    }};
}

DataProcessorSpec getClustersReaderSpec(bool propagateMC)
{

  auto initFunction = [propagateMC](InitContext& ic) {
    // get the option from the init context
    auto filename = ic.options().get<std::string>("infile");
    auto treename = ic.options().get<std::string>("treename");
    auto nofEvents = ic.options().get<int>("nevents");
    auto publishingMode = nofEvents == -1 ? RootTreeReader::PublishingMode::Single : RootTreeReader::PublishingMode::Loop;

    auto processAttributes = std::make_shared<ProcessAttributes>();
    {
      processAttributes->terminateOnEod = ic.options().get<bool>("terminate-on-eod");
      processAttributes->finished = false;
      processAttributes->datatype = "CPVCluster";
      constexpr auto persistency = Lifetime::Timeframe;
      o2::header::DataHeader::SubSpecificationType subSpec = 0;
      if (propagateMC) {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"CPV", "CLUSTERS", subSpec}, "CPVCluster"},
                                                                     RootTreeReader::BranchDefinition<std::vector<TriggerRecord>>{Output{"CPV", "CLUSTERTRIGRECS", subSpec}, "CPVClusterTrigRec"},
                                                                     RootTreeReader::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{Output{"CPV", "CLUSTERTRUEMC", subSpec}, "CPVClusterTrueMC"});
      } else {
        processAttributes->reader = std::make_shared<RootTreeReader>(treename.c_str(), // tree name
                                                                     filename.c_str(), // input file name
                                                                     nofEvents,        // number of entries to publish
                                                                     publishingMode,
                                                                     RootTreeReader::BranchDefinition<std::vector<Cluster>>{Output{"CPV", "CLUSTERS", subSpec}, "CPVCluster"},
                                                                     RootTreeReader::BranchDefinition<std::vector<TriggerRecord>>{Output{"CPV", "CLUSTERTRIGRECS", subSpec}, "CPVClusterTrigRec"});
      }
    }

    auto processFunction = [processAttributes, propagateMC](ProcessingContext& pc) {
      if (processAttributes->finished) {
        return;
      }

      auto publish = [&processAttributes, &pc, propagateMC]() {
        o2::cpv::CPVBlockHeader cpvheader(true);
        if (processAttributes->reader->next()) {
          (*processAttributes->reader)(pc, cpvheader);
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
        o2::cpv::CPVBlockHeader dummyheader(false);
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
  outputSpecs.emplace_back(OutputSpec{{"output"}, "CPV", "CLUSTERS", 0, Lifetime::Timeframe});
  outputSpecs.emplace_back(OutputSpec{{"outputTR"}, "CPV", "CLUSTERTRIGRECS", 0, Lifetime::Timeframe});
  if (propagateMC) {
    outputSpecs.emplace_back(OutputSpec{{"outputMC"}, "CPV", "CLUSTERTRUEMC", 0, Lifetime::Timeframe});
  }

  return DataProcessorSpec{
    "cpv-cluster-reader",
    Inputs{}, // no inputs
    outputSpecs,
    AlgorithmSpec(initFunction),
    Options{
      {"infile", VariantType::String, "cpvclusters.root", {"Name of the input file"}},
      {"treename", VariantType::String, "o2sim", {"Name of input tree"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"nevents", VariantType::Int, -1, {"number of events to run, -1: inf loop"}},
      {"terminate-on-eod", VariantType::Bool, true, {"terminate on end-of-data"}},
    }};
}

} // namespace cpv

} // namespace o2

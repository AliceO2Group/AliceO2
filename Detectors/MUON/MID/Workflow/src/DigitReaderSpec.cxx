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

/// \file   MID/Workflow/src/DigitReaderSpec.cxx
/// \brief  Data processor spec for MID digits reader device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 April 2019

#include "MIDWorkflow/DigitReaderSpec.h"

#include <memory>
#include <sstream>
#include <string>
#include "DPLUtils/RootTreeReader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/MCLabel.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace mid
{

class DigitsReaderDeviceDPL
{
 public:
  DigitsReaderDeviceDPL(bool useMC, const std::vector<header::DataDescription>& descriptions)
    : mUseMC(useMC), mDescriptions(descriptions) {}

  void init(InitContext& ic)
  {
    auto filename = utils::Str::concat_string(utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("mid-digit-infile"));
    if (mUseMC) {
      mReader = std::make_unique<RootTreeReader>("o2sim", filename.c_str(), -1,
                                                 RootTreeReader::PublishingMode::Single,
                                                 RootTreeReader::BranchDefinition<std::vector<ColumnData>>{
                                                   Output{header::gDataOriginMID, mDescriptions[0], 0}, "MIDDigit"},
                                                 RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{
                                                   Output{header::gDataOriginMID, mDescriptions[1], 0}, "MIDROFRecords"},
                                                 RootTreeReader::BranchDefinition<dataformats::MCTruthContainer<MCLabel>>{
                                                   Output{header::gDataOriginMID, mDescriptions[2], 0}, "MIDDigitMCLabels"},
                                                 &mPublishDigits);
    } else {
      mReader = std::make_unique<RootTreeReader>("o2sim", filename.c_str(), -1,
                                                 RootTreeReader::PublishingMode::Single,
                                                 RootTreeReader::BranchDefinition<std::vector<ColumnData>>{
                                                   Output{header::gDataOriginMID, mDescriptions[0], 0}, "MIDDigit"},
                                                 RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{
                                                   Output{header::gDataOriginMID, mDescriptions[1], 0}, "MIDROFRecords"},
                                                 &mPublishDigits);
    }
  }

  void run(ProcessingContext& pc)
  {
    if ((++(*mReader))(pc) == false) {
      pc.services().get<ControlService>().endOfStream();
    }
  }

 private:
  std::unique_ptr<RootTreeReader> mReader{};
  std::vector<header::DataDescription> mDescriptions{};
  bool mUseMC = true;

  /// structure holding the function to convert and publish the digits
  RootTreeReader::SpecialPublishHook mPublishDigits{
    [](std::string_view name, ProcessingContext& pc, Output const& output, char* data) -> bool {
      if (name == "MIDDigit") {
        auto inputDigits = reinterpret_cast<std::vector<ColumnData>*>(data);
        std::vector<ColumnData> digits{};
        digits.insert(digits.end(), inputDigits->begin(), inputDigits->end());
        pc.outputs().snapshot(output, digits);
        LOG(debug) << "MIDDigitsReader pushed " << digits.size() << " digits";
        return true;
      }
      return false;
    }};
};

DataProcessorSpec getDigitReaderSpec(bool useMC, const char* baseDescription)
{
  std::vector<OutputSpec> outputs;
  std::vector<header::DataDescription> descriptions;
  std::stringstream ss;
  ss << "A:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "/0";
  ss << ";B:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "ROF/0";
  if (useMC) {
    ss << ";C:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "LABELS/0";
  }
  auto matchers = select(ss.str().c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
    descriptions.emplace_back(DataSpecUtils::asConcreteDataDescription(matcher));
  }

  return DataProcessorSpec{
    "MIDDigitsReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitsReaderDeviceDPL>(useMC, descriptions)},
    Options{{"mid-digit-infile", VariantType::String, "middigits.root", {"Name of the input file"}},
            {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}
} // namespace mid
} // namespace o2

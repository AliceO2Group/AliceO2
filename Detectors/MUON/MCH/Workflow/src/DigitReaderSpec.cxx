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

/// \file   MCH/Workflow/src/DigitReaderSpec.cxx
/// \brief  Data processor spec for MCH digits reader device
/// \author Michael Winn <Michael.Winn at cern.ch>
/// \date   17 April 2021

#include "DigitReaderSpec.h"

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
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace mch
{

class DigitsReaderDeviceDPL
{
 public:
  DigitsReaderDeviceDPL(bool useMC, const std::vector<header::DataDescription>& descriptions)
    : mUseMC(useMC), mDescriptions(descriptions) {}

  void init(InitContext& ic)
  {
    auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")), ic.options().get<std::string>("mch-digit-infile"));
    if (mUseMC) {
      mReader = std::make_unique<RootTreeReader>("o2sim", filename.c_str(), -1,
                                                 RootTreeReader::PublishingMode::Single,
                                                 RootTreeReader::BranchDefinition<std::vector<Digit>>{
                                                   Output{header::gDataOriginMCH, mDescriptions[0], 0, Lifetime::Timeframe}, "MCHDigit"},
                                                 RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{
                                                   Output{header::gDataOriginMCH, mDescriptions[1], 0, Lifetime::Timeframe}, "MCHROFRecords"},
                                                 RootTreeReader::BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{
                                                   Output{header::gDataOriginMCH, mDescriptions[2], 0, Lifetime::Timeframe}, "MCHMCLabels"});
    } else {
      mReader = std::make_unique<RootTreeReader>("o2sim", filename.c_str(), -1,
                                                 RootTreeReader::PublishingMode::Single,
                                                 RootTreeReader::BranchDefinition<std::vector<Digit>>{
                                                   Output{header::gDataOriginMCH, mDescriptions[0], 0, Lifetime::Timeframe}, "MCHDigit"},
                                                 RootTreeReader::BranchDefinition<std::vector<ROFRecord>>{
                                                   Output{header::gDataOriginMCH, mDescriptions[1], 0, Lifetime::Timeframe}, "MCHROFRecords"});
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
};

framework::DataProcessorSpec getDigitReaderSpec(bool useMC, const char* specName)
{
  std::vector<OutputSpec> outputs;
  std::vector<header::DataDescription> descriptions;
  std::stringstream ss;
  ss << "A:" << header::gDataOriginMCH.as<std::string>() << "/DIGITS/0";
  ss << ";B:" << header::gDataOriginMCH.as<std::string>() << "/DIGITROFS/0";
  if (useMC) {
    ss << ";C:" << header::gDataOriginMCH.as<std::string>() << "/DIGITLABELS/0";
  }
  auto matchers = select(ss.str().c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(DataSpecUtils::asOutputSpec(matcher));
    descriptions.emplace_back(DataSpecUtils::asConcreteDataDescription(matcher));
  }

  return DataProcessorSpec{
    specName,
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<DigitsReaderDeviceDPL>(useMC, descriptions)},
    Options{{"mch-digit-infile", VariantType::String, "mchdigits.root", {"Name of the input file"}},
            {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}
} // namespace mch
} // namespace o2

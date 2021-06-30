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

#include <sstream>
#include <string>
#include "fmt/format.h"
#include "TFile.h"
#include "TTree.h"
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
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/MCLabel.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "CommonUtils/StringUtils.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class DigitsReaderDeviceDPL
{
 public:
  DigitsReaderDeviceDPL(bool useMC, const std::vector<header::DataDescription>& descriptions) : mUseMC(useMC), mDescriptions(descriptions) {}
  void init(o2::framework::InitContext& ic)
  {
    auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                  ic.options().get<std::string>("mid-digit-infile"));
    mFile = std::make_unique<TFile>(filename.c_str());
    if (!mFile->IsOpen()) {
      LOG(ERROR) << "Cannot open the " << filename << " file !";
      mState = 1;
      return;
    }
    mTree = static_cast<TTree*>(mFile->Get("o2sim"));
    if (!mTree) {
      LOG(ERROR) << "Cannot find tree in " << filename;
      mState = 1;
      return;
    }
    mTree->SetBranchAddress("MIDDigit", &mDigits);
    if (mUseMC) {
      mTree->SetBranchAddress("MIDDigitMCLabels", &mMCContainer);
    }
    mTree->SetBranchAddress("MIDROFRecords", &mROFRecords);
    mState = 0;
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    if (mState != 0) {
      return;
    }

    std::vector<ColumnData> digits;
    o2::dataformats::MCTruthContainer<MCLabel> mcContainer;
    std::vector<ROFRecord> rofRecords;

    for (auto ientry = 0; ientry < mTree->GetEntries(); ++ientry) {
      mTree->GetEntry(ientry);
      digits.insert(digits.end(), mDigits->begin(), mDigits->end());
      rofRecords.insert(rofRecords.end(), mROFRecords->begin(), mROFRecords->end());
      if (mUseMC) {
        mcContainer.mergeAtBack(*mMCContainer);
      }
    }

    LOG(DEBUG) << "MIDDigitsReader pushed " << digits.size() << " merged digits";
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, mDescriptions[0], 0, of::Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, mDescriptions[1], 0, of::Lifetime::Timeframe}, rofRecords);
    LOG(DEBUG) << "MIDDigitsReader pushed " << digits.size() << " indexed digits";
    if (mUseMC) {
      pc.outputs().snapshot(of::Output{header::gDataOriginMID, mDescriptions[2], 0, of::Lifetime::Timeframe}, mcContainer);
    }
    mState = 2;
    pc.services().get<of::ControlService>().endOfStream();
  }

 private:
  std::unique_ptr<TFile> mFile{nullptr};
  TTree* mTree{nullptr};                                             // not owner
  std::vector<o2::mid::ColumnDataMC>* mDigits{nullptr};              // not owner
  o2::dataformats::MCTruthContainer<MCLabel>* mMCContainer{nullptr}; // not owner
  std::vector<o2::mid::ROFRecord>* mROFRecords{nullptr};             // not owner
  std::vector<header::DataDescription> mDescriptions{};
  int mState = 0;
  bool mUseMC = true;
};

framework::DataProcessorSpec getDigitReaderSpec(bool useMC, const char* baseDescription)
{
  std::vector<of::OutputSpec> outputs;
  std::vector<header::DataDescription> descriptions;
  std::stringstream ss;
  ss << "A:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "/0";
  ss << ";B:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "ROF/0";
  if (useMC) {
    ss << ";C:" << header::gDataOriginMID.as<std::string>() << "/" << baseDescription << "LABELS/0";
  }
  auto matchers = of::select(ss.str().c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(of::DataSpecUtils::asOutputSpec(matcher));
    descriptions.emplace_back(of::DataSpecUtils::asConcreteDataDescription(matcher));
  }

  return of::DataProcessorSpec{
    "MIDDigitsReader",
    of::Inputs{},
    outputs,
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::DigitsReaderDeviceDPL>(useMC, descriptions)},
    of::Options{{"mid-digit-infile", of::VariantType::String, "middigits.root", {"Name of the input file"}},
                {"input-dir", of::VariantType::String, "none", {"Input directory"}}}};
}
} // namespace mid
} // namespace o2

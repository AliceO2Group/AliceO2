// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MCH/Workflow/src/DigitReaderSpec.cxx
/// \brief  Data processor spec for MCH digits reader device
/// \author Michael Winn <Michael.Winn at cern.ch>
/// \date   17 April 2021

#include "DigitReaderSpec.h"

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
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"

namespace of = o2::framework;

namespace o2
{
namespace mch
{

class DigitsReaderDeviceDPL
{
 public:
  DigitsReaderDeviceDPL(bool useMC, const std::vector<header::DataDescription>& descriptions) : mUseMC(useMC), mDescriptions(descriptions) {}
  void init(o2::framework::InitContext& ic)
  {
    auto filename = ic.options().get<std::string>("mch-digit-infile");
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
    mTree->SetBranchAddress("MCHDigit", &mDigits);
    if (mUseMC) {
      mTree->SetBranchAddress("MCHMCLabels", &mMCContainer);
    }
    mTree->SetBranchAddress("MCHROFRecords", &mROFRecords);
    mState = 0;
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    if (mState != 0) {
      return;
    }

    std::vector<o2::mch::Digit> digits;
    o2::dataformats::MCTruthContainer<MCCompLabel> mcContainer;
    std::vector<ROFRecord> rofRecords;

    for (auto ientry = 0; ientry < mTree->GetEntries(); ++ientry) {
      mTree->GetEntry(ientry);
      digits.insert(digits.end(), mDigits->begin(), mDigits->end());
      rofRecords.insert(rofRecords.end(), mROFRecords->begin(), mROFRecords->end());
      if (mUseMC) {
        mcContainer.mergeAtBack(*mMCContainer);
      }
    }

    LOG(DEBUG) << "MCHDigitsReader pushed " << digits.size() << " merged digits";
    pc.outputs().snapshot(of::Output{header::gDataOriginMCH, mDescriptions[0], 0, of::Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(of::Output{header::gDataOriginMCH, mDescriptions[1], 0, of::Lifetime::Timeframe}, rofRecords);
    LOG(DEBUG) << "MCHDigitsReader pushed " << digits.size() << " indexed digits";
    if (mUseMC) {
      pc.outputs().snapshot(of::Output{header::gDataOriginMCH, mDescriptions[2], 0, of::Lifetime::Timeframe}, mcContainer);
    }
    mState = 2;
    pc.services().get<of::ControlService>().endOfStream();
  }

 private:
  std::unique_ptr<TFile> mFile{nullptr};
  TTree* mTree{nullptr};                                                 // not owner
  std::vector<o2::mch::Digit>* mDigits{nullptr};                         // not owner
  o2::dataformats::MCTruthContainer<MCCompLabel>* mMCContainer{nullptr}; // not owner
  std::vector<o2::mch::ROFRecord>* mROFRecords{nullptr};                 // not owner
  std::vector<header::DataDescription> mDescriptions{};
  int mState = 0;
  bool mUseMC = true;
};

framework::DataProcessorSpec getDigitReaderSpec(bool useMC, const char* baseDescription)
{
  std::vector<of::OutputSpec> outputs;
  std::vector<header::DataDescription> descriptions;
  std::stringstream ss;
  ss << "A:" << header::gDataOriginMCH.as<std::string>() << "/" << baseDescription << "S/0";
  ss << ";B:" << header::gDataOriginMCH.as<std::string>() << "/" << baseDescription << "ROFS/0";
  if (useMC) {
    ss << ";C:" << header::gDataOriginMCH.as<std::string>() << "/" << baseDescription << "LABELS/0";
  }
  auto matchers = of::select(ss.str().c_str());
  for (auto& matcher : matchers) {
    outputs.emplace_back(of::DataSpecUtils::asOutputSpec(matcher));
    descriptions.emplace_back(of::DataSpecUtils::asConcreteDataDescription(matcher));
  }

  return of::DataProcessorSpec{
    "MCHDigitsReader",
    of::Inputs{},
    outputs,
    of::AlgorithmSpec{of::adaptFromTask<o2::mch::DigitsReaderDeviceDPL>(useMC, descriptions)},
    of::Options{{"mch-digit-infile", of::VariantType::String, "mchdigits.root", {"Name of the input file"}}}};
}
} // namespace mch
} // namespace o2

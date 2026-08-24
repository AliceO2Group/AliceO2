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

#include "TFile.h"
#include "TTree.h"
#include "DataFormatsCTP/Digits.h"
#include "DataFormatsCTP/LumiInfo.h"
#include "Headers/DataHeader.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/IRFrameSelector.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace ctp
{

class DigitReader : public Task
{
 public:
  DigitReader() = delete;
  DigitReader(bool useMC);
  ~DigitReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::vector<o2::ctp::CTPDigit> mDigits, *mDigitsPtr = &mDigits;
  o2::ctp::LumiInfo mLumi, *mLumiPtr = &mLumi;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  bool mUseMC = false; // use MC truth
  bool mUseIRFrames = false; // selected IRFrames mode
  std::string mDigTreeName = "o2sim";
  std::string mDigitBranchName = "CTPDigits";
  std::string mLumiBranchName = "CTPLumi";
};

DigitReader::DigitReader(bool useMC)
{
  if (useMC) {
    LOG(info) << "CTP : truth = data as CTP inputs are already digital";
  }
}

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("ctp-digit-infile"));
  if (ic.options().hasOption("ignore-irframes") && !ic.options().get<bool>("ignore-irframes")) {
    mUseIRFrames = true;
  }
  connectTree(filename);
}

void DigitReader::run(ProcessingContext& pc)
{
  gsl::span<const o2::dataformats::IRFrame> irFrames{};
  // LOG(info) << "Using IRs:" << mUseIRFrames;
  if (mUseIRFrames) {
    irFrames = pc.inputs().get<gsl::span<o2::dataformats::IRFrame>>("driverInfo");
  }
  auto ent = mTree->GetReadEntry();
  if (!mUseIRFrames) {
    ent++;
    if (ent >= mTree->GetEntries()) {
      // A timeframe holds no collision at all whenever the interaction rate is low enough, and
      // the tree then has no entry to read. End the stream instead of reading past the end and
      // publishing branch addresses that GetEntry has not filled. This was an assert, which is
      // compiled out of every production build since ENABLE_CASSERT defaults to OFF.
      LOG(info) << "no entry to read, ending the stream";
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
      return;
    }
    mTree->GetEntry(ent);
    LOG(info) << "DigitReader pushes " << mDigits.size() << " digits at entry " << ent;
    pc.outputs().snapshot(Output{"CTP", "DIGITS", 0}, mDigits);
    pc.outputs().snapshot(Output{"CTP", "LUMI", 0}, mLumi);
    if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  } else {
    std::vector<o2::ctp::CTPDigit> digitSel;
    if (irFrames.size()) { // we assume the IRFrames are in the increasing order
      if (ent < 0) {
        ent++;
      }
      o2::utils::IRFrameSelector irfSel;
      // MC  digits are already aligned
      irfSel.setSelectedIRFrames(irFrames, 0, 0, 0, true);
      const auto irMin = irfSel.getIRFrames().front().getMin(); // use processed IRframes for rough comparisons (possible shift!)
      const auto irMax = irfSel.getIRFrames().back().getMax();
      LOGP(info, "Selecting IRFrame {}-{}", irMin.asString(), irMax.asString());
      while (ent < mTree->GetEntries()) {
        if (ent > mTree->GetReadEntry()) {
          mTree->GetEntry(ent);
        }
        if (mDigits.front().intRecord <= irMax && mDigits.back().intRecord >= irMin) { // THere is overlap
          for (int i = 0; i < (int)mDigits.size(); i++) {
            const auto& dig = mDigits[i];
            // if(irfSel.check(dig.intRecord)) { // adding selected digit
            if (dig.intRecord >= irMin && dig.intRecord <= irMax) {
              digitSel.push_back(dig);
              LOG(info) << "adding:" << dig.intRecord << " ent:" << ent;
            }
          }
        }
        if (mDigits.back().intRecord < irMax) { // need to check the next entry
          ent++;
          continue;
        }
        break; // push collected data
      }
    }
    pc.outputs().snapshot(Output{"CTP", "DIGITS", 0}, digitSel);
    pc.outputs().snapshot(Output{"CTP", "LUMI", 0}, mLumi); // add full lumi for this TF
    if (!irFrames.size() || irFrames.back().isLast()) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }
}

void DigitReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mDigTreeName.c_str()));
  assert(mTree);
  if (mTree->GetBranch(mDigitBranchName.c_str())) {
    mTree->SetBranchAddress(mDigitBranchName.c_str(), &mDigitsPtr);
  } else {
    LOGP(warn, "Digits branch {} is absent", mDigitBranchName);
  }
  if (mTree->GetBranch(mLumiBranchName.c_str())) {
    mTree->SetBranchAddress(mLumiBranchName.c_str(), &mLumiPtr);
  } else {
    LOGP(warn, "Lumi branch {} is absent", mLumiBranchName);
  }
  mTree->SetBranchAddress(mDigitBranchName.c_str(), &mDigitsPtr);
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getDigitsReaderSpec(bool useMC, const std::string& defFile)
{
  return DataProcessorSpec{
    "ctp-digit-reader",
    Inputs{},
    Outputs{{"CTP", "DIGITS", 0, Lifetime::Timeframe},
            {"CTP", "LUMI", 0, o2::framework::Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC)},
    Options{
      {"ctp-digit-infile", VariantType::String, defFile, {"Name of the input digit file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace ctp

} // namespace o2

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
#include "CommonUtils/NameConf.h"
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
  std::string mDigTreeName = "o2sim";
  std::string mDigitBranchName = "CTPDigits";
  std::string mLumiBranchName = "CTPLumi";
};

DigitReader::DigitReader(bool useMC)
{
  if (useMC) {
    LOG(info) << "CTP does not support MC truth at the moment";
  }
}

void DigitReader::init(InitContext& ic)
{
  auto filename = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                ic.options().get<std::string>("ctp-digit-infile"));
  connectTree(filename);
}

void DigitReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen

  mTree->GetEntry(ent);
  LOG(info) << "DigitReader pushes " << mDigits.size() << " digits at entry " << ent;
  pc.outputs().snapshot(Output{"CTP", "DIGITS", 0, Lifetime::Timeframe}, mDigits);
  pc.outputs().snapshot(Output{"CTP", "LUMI", 0, Lifetime::Timeframe}, mLumi);
  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
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

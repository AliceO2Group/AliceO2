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

/// @file   DigitReaderSpec.cxx

#include <vector>
#include <cassert>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "PHOSWorkflow/DigitReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::phos;

namespace o2
{
namespace phos
{

DigitReader::DigitReader(bool useMC)
{
  mUseMC = useMC;
}

void DigitReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("phos-digits-infile"));
  connectTree(mInputFileName);
}

void DigitReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mDigits.size() << " Digits in " << mTRs.size() << " TriggerRecords at entry " << ent;
  pc.outputs().snapshot(Output{mOrigin, "DIGITS", 0, Lifetime::Timeframe}, mDigits);
  pc.outputs().snapshot(Output{mOrigin, "DIGITTRIGREC", 0, Lifetime::Timeframe}, mTRs);
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "DIGITSMCTR", 0, Lifetime::Timeframe}, mMCTruth);
  }

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
  mTree.reset((TTree*)mFile->Get(mDigitTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mTRBranchName.c_str()));

  mTree->SetBranchAddress(mTRBranchName.c_str(), &mTRsInp);
  mTree->SetBranchAddress(mDigitBranchName.c_str(), &mDigitsInp);
  if (mUseMC) {
    if (mTree->GetBranch(mDigitMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mDigitMCTruthBranchName.c_str(), &mMCTruthInp);
    } else {
      LOG(warning) << "MC-truth is missing, message will be empty";
    }
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getPHOSDigitReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("PHS", "DIGITS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("PHS", "DIGITTRIGREC", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("PHS", "DIGITSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "phos-digits-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<DigitReader>(useMC)},
    Options{
      {"phos-digits-infile", VariantType::String, "phosdigits.root", {"Name of the input Digit file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace phos
} // namespace o2

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

/// @file   CellReaderSpec.cxx

#include <vector>
#include <cassert>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "PHOSWorkflow/CellReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::phos;

namespace o2
{
namespace phos
{

CellReader::CellReader(bool useMC)
{
  mUseMC = useMC;
}

void CellReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("phos-cells-infile"));
  connectTree(mInputFileName);
}

void CellReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mCells.size() << " Cells in " << mTRs.size() << " TriggerRecords at entry " << ent;
  pc.outputs().snapshot(Output{mOrigin, "CELLS", 0, Lifetime::Timeframe}, mCells);
  pc.outputs().snapshot(Output{mOrigin, "CELLTRIGREC", 0, Lifetime::Timeframe}, mTRs);
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "CELLSMCTR", 0, Lifetime::Timeframe}, mMCTruth);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void CellReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mCellTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mTRBranchName.c_str()));

  mTree->SetBranchAddress(mTRBranchName.c_str(), &mTRsInp);
  mTree->SetBranchAddress(mCellBranchName.c_str(), &mCellsInp);
  if (mUseMC) {
    if (mTree->GetBranch(mCellMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mCellMCTruthBranchName.c_str(), &mMCTruthInp);
    } else {
      LOG(warning) << "MC-truth is missing, message will be empty";
    }
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getPHOSCellReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("PHS", "CELLS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("PHS", "CELLTRIGREC", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("PHS", "CELLSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "phos-cells-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<CellReader>(useMC)},
    Options{
      {"phos-cells-infile", VariantType::String, "phoscells.root", {"Name of the input Cell file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace phos
} // namespace o2

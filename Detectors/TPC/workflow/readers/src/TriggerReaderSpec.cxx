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

/// @file   TriggerReaderSpec.cxx

#include <vector>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCReaderWorkflow/TriggerReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

void TriggerReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("infile"));
  connectTree(mInputFileName);
}

void TriggerReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  mTree->GetEntry(ent);

  pc.outputs().snapshot(Output{"TPC", "TRIGGERWORDS", 0, Lifetime::Timeframe}, *mTrig);
  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TriggerReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  if (!(mFile && !mFile->IsZombie())) {
    throw std::runtime_error("Error opening tree file");
  }
  mTree.reset((TTree*)mFile->Get(mTriggerTreeName.c_str()));
  if (!mTree) {
    throw std::runtime_error("Error opening tree");
  }

  mTree->SetBranchAddress(mTriggerBranchName.c_str(), &mTrig);
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getTPCTriggerReaderSpec()
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("TPC", "TRIGGERWORDS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "tpc-trigger-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<TriggerReader>()},
    Options{
      {"infile", VariantType::String, "tpctriggers.root", {"Name of the input triggers file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace tpc
} // namespace o2

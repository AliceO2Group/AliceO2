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

/// @file   RecEventReaderSpec.cxx

#include <vector>

#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "ZDCWorkflow/RecEventReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::zdc;

namespace o2
{
namespace zdc
{

RecEventReader::RecEventReader(bool useMC)
{
  mUseMC = useMC;
  if (useMC) {
    LOG(warning) << "ZDC RecEvent reader at the moment does not process MC";
  }
}

void RecEventReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("zdc-reco-infile"));
  connectTree(mInputFileName);
}

void RecEventReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);

  LOG(info) << "ZDC RecEventReader pushes " << mBCRecData->size() << " events with " << mBCRecData->size() << " energy, " << mZDCTDCData->size() << " TDC and " << mZDCInfo->size() << " info records at entry " << ent;
  pc.outputs().snapshot(Output{"ZDC", "BCREC", 0}, *mBCRecData);
  pc.outputs().snapshot(Output{"ZDC", "ENERGY", 0}, *mZDCEnergy);
  pc.outputs().snapshot(Output{"ZDC", "TDCDATA", 0}, *mZDCTDCData);
  pc.outputs().snapshot(Output{"ZDC", "INFO", 0}, *mZDCInfo);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void RecEventReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded

  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mRecEventTreeName.c_str()));
  assert(mTree);

  mTree->SetBranchAddress(mBCRecDataBranchName.c_str(), &mBCRecData);
  mTree->SetBranchAddress(mZDCEnergyBranchName.c_str(), &mZDCEnergy);
  mTree->SetBranchAddress(mZDCTDCDataBranchName.c_str(), &mZDCTDCData);
  mTree->SetBranchAddress(mZDCInfoBranchName.c_str(), &mZDCInfo);
  if (mUseMC) {
    LOG(warning) << "MC-truth is not supported for ZDC recpoints currently";
    mUseMC = false;
  }

  LOG(info) << "Loaded ZDC RecEvents tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getRecEventReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ZDC", "BCREC", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "ENERGY", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "TDCDATA", 0, Lifetime::Timeframe);
  outputs.emplace_back("ZDC", "INFO", 0, Lifetime::Timeframe);
  if (useMC) {
    LOG(warning) << "MC-truth is not supported for ZDC RecEvents currently";
  }

  return DataProcessorSpec{
    "zdc-reco-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<RecEventReader>()},
    Options{
      {"zdc-reco-infile", VariantType::String, "zdcreco.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace zdc
} // namespace o2

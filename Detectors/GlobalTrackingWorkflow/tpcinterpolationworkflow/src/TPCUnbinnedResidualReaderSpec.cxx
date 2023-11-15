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

/// @file   TPCUnbinnedResidualReaderSpec.cxx

#include "TPCInterpolationWorkflow/TPCUnbinnedResidualReaderSpec.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "fairlogger/Logger.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

void TPCUnbinnedResidualReader::init(InitContext& ic)
{
  // get the option from the init context
  LOG(info) << "Init TPC unbinned residual reader!";
  mInFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                              ic.options().get<std::string>("input-filename"));
  mInTreeName = ic.options().get<std::string>("treename");
  connectTree();
}

void TPCUnbinnedResidualReader::connectTree()
{
  mTreeIn.reset(nullptr); // in case it was already loaded
  mFileIn.reset(TFile::Open(mInFileName.c_str()));
  assert(mFileIn && !mFileIn->IsZombie());
  mTreeIn.reset((TTree*)mFileIn->Get(mInTreeName.c_str()));
  assert(mTreeIn);
  mTreeIn->SetBranchAddress("residuals", &mUnbinnedResidPtr);
  mTreeIn->SetBranchAddress("trackRefs", &mTrackDataCompactPtr);
  if (mTrackInput) {
    mTreeIn->SetBranchAddress("tracks", &mTrackDataPtr);
  }
  LOG(info) << "Loaded tree from " << mInFileName << " with " << mTreeIn->GetEntries() << " entries";
}

void TPCUnbinnedResidualReader::run(ProcessingContext& pc)
{
  auto currEntry = mTreeIn->GetReadEntry() + 1;
  assert(currEntry < mTreeIn->GetEntries()); // this should not happen
  mTreeIn->GetEntry(currEntry);
  LOG(info) << "Pushing " << mUnbinnedResid.size() << " unbinned residuals at entry " << currEntry;
  pc.outputs().snapshot(Output{"GLO", "UNBINNEDRES", 0, Lifetime::Timeframe}, mUnbinnedResid);
  pc.outputs().snapshot(Output{"GLO", "TRKREFS", 0, Lifetime::Timeframe}, mTrackDataCompact);
  if (mTrackInput) {
    LOG(info) << "Pushing " << mTrackData.size() << " reference tracks for these residuals";
    pc.outputs().snapshot(Output{"GLO", "TRKDATA", 0, Lifetime::Timeframe}, mTrackData);
  }

  if (mTreeIn->GetReadEntry() + 1 >= mTreeIn->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

DataProcessorSpec getUnbinnedTPCResidualsReaderSpec(bool trkInput)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "UNBINNEDRES", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "TRKREFS", 0, Lifetime::Timeframe);
  if (trkInput) {
    outputs.emplace_back("GLO", "TRKDATA", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "TPCUnbinnedResidualReader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<TPCUnbinnedResidualReader>(trkInput)},
    Options{
      {"input-filename", VariantType::String, "o2residuals_tpc.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"treename", VariantType::String, "residualsTPC", {"Name of top-level TTree"}},
    }};
}

} // namespace tpc
} // namespace o2

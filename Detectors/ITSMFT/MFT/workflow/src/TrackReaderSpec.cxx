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

/// @file   TrackReaderSpec.cxx

#include <vector>
#include <cassert>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "MFTWorkflow/TrackReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::mft;

namespace o2
{
namespace mft
{

TrackReader::TrackReader(bool useMC)
{
  mUseMC = useMC;
}

void TrackReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("mft-tracks-infile"));
  connectTree(mInputFileName);
}

void TrackReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mTracks.size() << " track in " << mROFRec.size() << " ROFs at entry " << ent;
  pc.outputs().snapshot(Output{mOrigin, "MFTTrackROF", 0}, mROFRec);
  pc.outputs().snapshot(Output{mOrigin, "TRACKS", 0}, mTracks);
  pc.outputs().snapshot(Output{mOrigin, "TRACKCLSID", 0}, mClusInd);
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "TRACKSMCTR", 0}, mMCTruth);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TrackReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mTrackTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mROFBranchName.c_str()));

  mTree->SetBranchAddress(mROFBranchName.c_str(), &mROFRecInp);
  mTree->SetBranchAddress(mTrackBranchName.c_str(), &mTracksInp);
  mTree->SetBranchAddress(mClusIdxBranchName.c_str(), &mClusIndInp);

  if (mUseMC) {
    if (mTree->GetBranch(mTrackMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mTrackMCTruthBranchName.c_str(), &mMCTruthInp);
    } else {
      LOG(warning) << "MC-truth is missing, message will be empty";
    }
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getMFTTrackReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("MFT", "MFTTrackROF", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("MFT", "TRACKS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("MFT", "TRACKCLSID", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("MFT", "TRACKSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-track-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<TrackReader>(useMC)},
    Options{
      {"mft-tracks-infile", VariantType::String, "mfttracks.root", {"Name of the input track file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace mft
} // namespace o2

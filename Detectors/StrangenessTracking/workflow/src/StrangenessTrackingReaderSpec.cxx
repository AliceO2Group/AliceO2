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

/// @file   StrangenessTrackingReaderSpec.cxx

#include "StrangenessTrackingWorkflow/StrangenessTrackingReaderSpec.h"

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "CommonUtils/NameConf.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/StrangeTrack.h"
#include "TFile.h"
#include "TTree.h"

using namespace o2::framework;

namespace o2
{
namespace strangeness_tracking
{
// read strangeness tracking candidates
class StrangenessTrackingReader : public o2::framework::Task
{
  // using RRef = o2::dataformats::RangeReference<int, int>;
  using StrangeTrack = dataformats::StrangeTrack;

 public:
  StrangenessTrackingReader() = default;
  ~StrangenessTrackingReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree();

  bool mVerbose = false;

  std::vector<StrangeTrack> mStrangeTrack, *mStrangeTrackPtr = &mStrangeTrack;
  // std::vector<RRef> mPV2V0Ref, *mPV2V0RefPtr = &mPV2V0Ref;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::string mFileNameMatches = "";
  std::string mSTrackingTreeName = "o2sim";
  std::string mStrackBranchName = "StrangeTracks";
  // std::string mPVertex2V0RefBranchName = "PV2V0Refs";
};

void StrangenessTrackingReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("secondary-vertex-infile"));
  connectTree();
}

void StrangenessTrackingReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mStrangeTrack.size() << " strange tracks at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "STRANGETRACKS", 0, Lifetime::Timeframe}, mStrangeTrack);
  // pc.outputs().snapshot(Output{"GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe}, mPV2V0Ref);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void StrangenessTrackingReader::connectTree()
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mFileName.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mSTrackingTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mStrackBranchName.c_str()));
  // assert(mTree->GetBranch(mPVertex2V0RefBranchName.c_str()));

  mTree->SetBranchAddress(mStrackBranchName.c_str(), &mStrangeTrackPtr);
  // mTree->SetBranchAddress(mPVertex2V0RefBranchName.c_str(), &mPV2V0RefPtr);

  LOG(info) << "Loaded " << mSTrackingTreeName << " tree from " << mFileName << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getStrangenessTrackingReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "STRANGETRACKS", 0, Lifetime::Timeframe);           // found strange tracks
  // outputs.emplace_back("GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);   // prim.vertex -> V0s refs

  return DataProcessorSpec{
    "strangeness-tracking-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<StrangenessTrackingReader>()},
    Options{
      {"strange-tracks-infile", VariantType::String, "o2_strange_tracks.root", {"Name of the input strange tracks file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace vertexing
} // namespace o2

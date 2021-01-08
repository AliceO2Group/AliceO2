// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackReaderSpec.cxx

#include <vector>
#include <cassert>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/TrackReaderSpec.h"

using namespace o2::framework;
using namespace o2::its;

namespace o2
{
namespace its
{

TrackReader::TrackReader(bool useMC)
{
  mUseMC = useMC;
}

void TrackReader::init(InitContext& ic)
{
  mInputFileName = ic.options().get<std::string>("its-tracks-infile");
  connectTree(mInputFileName);
}

void TrackReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "Pushing " << mTracks.size() << " track in " << mROFRec.size() << " ROFs at entry " << ent;
  pc.outputs().snapshot(Output{mOrigin, "ITSTrackROF", 0, Lifetime::Timeframe}, mROFRec);
  pc.outputs().snapshot(Output{mOrigin, "TRACKS", 0, Lifetime::Timeframe}, mTracks);
  pc.outputs().snapshot(Output{mOrigin, "TRACKCLSID", 0, Lifetime::Timeframe}, mClusInd);
  pc.outputs().snapshot(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe}, mVertices);
  pc.outputs().snapshot(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe}, mVerticesROFRec);
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "TRACKSMCTR", 0, Lifetime::Timeframe}, mMCTruth);
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
  if (!mTree->GetBranch(mVertexBranchName.c_str())) {
    LOG(WARNING) << "No " << mVertexBranchName << " branch in " << mTrackTreeName << " -> vertices will be empty";
  } else {
    mTree->SetBranchAddress(mVertexBranchName.c_str(), &mVerticesInp);
  }
  if (!mTree->GetBranch(mVertexROFBranchName.c_str())) {
    LOG(WARNING) << "No " << mVertexROFBranchName << " branch in " << mTrackTreeName
                 << " -> vertices ROFrecords will be empty";
  } else {
    mTree->SetBranchAddress(mVertexROFBranchName.c_str(), &mVerticesROFRecInp);
  }
  if (mUseMC) {
    if (mTree->GetBranch(mTrackMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mTrackMCTruthBranchName.c_str(), &mMCTruthInp);
    } else {
      LOG(WARNING) << "MC-truth is missing, message will be empty";
    }
  }
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getITSTrackReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "VERTICESROF", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-track-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<TrackReader>(useMC)},
    Options{
      {"its-tracks-infile", VariantType::String, "o2trac_its.root", {"Name of the input track file"}}}};
}

} // namespace its
} // namespace o2

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

#include "TTree.h"

#include "Framework/ControlService.h"
#include "TPCWorkflow/TrackReaderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{

TrackReader::TrackReader(bool useMC)
{
  mUseMC = useMC;
}

void TrackReader::init(InitContext& ic)
{
  mInputFileName = ic.options().get<std::string>("tpc-tracks-infile");
}

void TrackReader::run(ProcessingContext& pc)
{

  if (mFinished) {
    return;
  }
  accumulate();

  LOG(INFO) << "TPCTrackReader pushes " << mTracksOut.size() << " tracks";
  pc.outputs().snapshot(Output{"TPC", "TRACKS", 0, Lifetime::Timeframe}, mTracksOut);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe}, mMCTruthOut);
  }

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

void TrackReader::accumulate()
{
  // load data from files
  TFile trFile(mInputFileName.c_str(), "read");
  if (trFile.IsZombie()) {
    LOG(FATAL) << "Failed to open tracks file " << mInputFileName;
  }
  TTree* trTree = (TTree*)trFile.Get(mTrackTreeName.c_str());
  if (!trTree) {
    LOG(FATAL) << "Failed to load tracks tree " << mTrackTreeName << " from " << mInputFileName;
  }
  LOG(INFO) << "Loaded tracks tree " << mTrackTreeName << " from " << mInputFileName;

  trTree->SetBranchAddress(mTrackBranchName.c_str(), &mTracksInp);
  if (mUseMC) {
    if (trTree->GetBranch(mTrackMCTruthBranchName.c_str())) {
      trTree->SetBranchAddress(mTrackMCTruthBranchName.c_str(), &mMCTruthInp);
      LOG(INFO) << "Will use MC-truth from " << mTrackMCTruthBranchName;
    } else {
      LOG(INFO) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  int nEnt = trTree->GetEntries();
  if (nEnt == 1) {
    trTree->GetEntry(0);
    mTracksOut.swap(*mTracksInp);
    if (mUseMC) {
      mMCTruthOut.mergeAtBack(*mMCTruthInp);
    }
  } else {
    int lastEntry = -1;
    int ntrAcc = 0;
    for (int iev = 0; iev < nEnt; iev++) {
      trTree->GetEntry(0);
      auto tr0 = mTracksInp->begin();
      auto tr1 = mTracksInp->end();
      std::copy(tr0, tr1, std::back_inserter(mTracksOut));
      // MC
      if (mUseMC) {
        mMCTruthOut.mergeAtBack(*mMCTruthInp);
      }
    }
  }
}

DataProcessorSpec getTPCTrackReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("TPC", "TRACKS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-track-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<TrackReader>(useMC)},
    Options{
      {"tpc-tracks-infile", VariantType::String, "tpctracks.root", {"Name of the input track file"}}}};
}

} // namespace tpc
} // namespace o2

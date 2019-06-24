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
}

void TrackReader::run(ProcessingContext& pc)
{

  if (mFinished) {
    return;
  }
  accumulate();

  LOG(INFO) << "ITSTrackReader pushes " << mROFRecOut.size() << " ROFRecords,"
            << mTracksOut.size() << " tracks";
  pc.outputs().snapshot(Output{ mOrigin, "ITSTrackROF", 0, Lifetime::Timeframe }, mROFRecOut);
  pc.outputs().snapshot(Output{ mOrigin, "TRACKS", 0, Lifetime::Timeframe }, mTracksOut);
  pc.outputs().snapshot(Output{ mOrigin, "TRACKCLSID", 0, Lifetime::Timeframe }, mClusIndOut);
  if (mUseMC) {
    pc.outputs().snapshot(Output{ mOrigin, "TRACKSMCTR", 0, Lifetime::Timeframe }, mMCTruthOut);
  }

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(false);
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
  TTree* rofTree = (TTree*)trFile.Get(mROFTreeName.c_str());
  if (!rofTree) {
    LOG(FATAL) << "Failed to load tracks ROF tree " << rofTree << " from " << mInputFileName;
  }
  LOG(INFO) << "Loaded tracks tree " << mTrackTreeName << " and ROFRecords " << mROFTreeName
            << " from " << mInputFileName;

  rofTree->SetBranchAddress(mROFTreeName.c_str(), &mROFRecInp);
  trTree->SetBranchAddress(mTrackBranchName.c_str(), &mTracksInp);
  trTree->SetBranchAddress(mClusIdxBranchName.c_str(), &mClusIndInp);
  if (mUseMC) {
    if (trTree->GetBranch(mTrackMCTruthBranchName.c_str())) {
      trTree->SetBranchAddress(mTrackMCTruthBranchName.c_str(), &mMCTruthInp);
      LOG(INFO) << "Will use MC-truth from " << mTrackMCTruthBranchName;
    } else {
      LOG(INFO) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  // it is possible that the track data is stored in multiple entries, in this case we need to refill to 1 single vector
  if (rofTree->GetEntries() > 1) {
    LOG(FATAL) << "Tracks ROFRecords tree has " << rofTree->GetEntries() << " entries instead of 1";
  }
  rofTree->GetEntry(0);
  int nROFs = mROFRecInp->size();
  mROFRecOut.swap(*mROFRecInp);
  int nEnt = trTree->GetEntries();
  if (nEnt == 1) {
    trTree->GetEntry(0);
    mTracksOut.swap(*mTracksInp);
    mClusIndOut.swap(*mClusIndInp);
    if (mUseMC) {
      mMCTruthOut.mergeAtBack(*mMCTruthInp);
    }
  } else {
    int lastEntry = -1;
    int ntrAcc = 0;
    for (auto& rof : mROFRecOut) {
      auto rEntry = rof.getROFEntry().getEvent();
      if (lastEntry != rEntry) {
        trTree->GetEntry((lastEntry = rEntry));
        int shiftIdx = mClusIndOut.size();
        std::copy(mClusIndInp->begin(), mClusIndInp->end(), std::back_inserter(mClusIndOut)); // cluster indices
        // update cluster indices
        if (shiftIdx) {
          for (auto& trc : *mTracksInp) {
            trc.shiftFirstClusterEntry(shiftIdx);
          }
        }
      }

      auto tr0 = mTracksInp->begin() + rof.getROFEntry().getIndex();
      auto tr1 = tr0 + rof.getNROFEntries();
      std::copy(tr0, tr1, std::back_inserter(mTracksOut));
      // MC
      if (mUseMC) {
        mMCTruthOut.mergeAtBack(*mMCTruthInp);
      }
      rof.getROFEntry().setEvent(0);
      rof.getROFEntry().setIndex(ntrAcc);
      ntrAcc += rof.getNROFEntries();
    }
  }
  delete trTree;
  delete rofTree;
  trFile.Close();
}

DataProcessorSpec getITSTrackReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-track-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{ adaptFromTask<TrackReader>(useMC) },
    Options{
      { "its-tracks-infile", VariantType::String, "o2trac_its.root", { "Name of the input track file" } } }
  };
}

} // namespace its
} // namespace o2

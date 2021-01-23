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
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
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
  mInputFileName = ic.options().get<std::string>("infile");
  connectTree(mInputFileName);
}

void TrackReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  accumulate(ent, 1);                // to really accumulate all, use accumulate(ent,mTree->GetEntries());
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);

  pc.outputs().snapshot(Output{"TPC", "TRACKS", 0, Lifetime::Timeframe}, mTracksOut);
  pc.outputs().snapshot(Output{"TPC", "CLUSREFS", 0, Lifetime::Timeframe}, mCluRefVecOut);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe}, mMCTruthOut);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void TrackReader::accumulate(int from, int n)
{
  assert(from + n <= mTree->GetEntries());
  if (n == 1) {
    mTree->GetEntry(from);
    mTracksOut.swap(*mTracksInp);
    mCluRefVecOut.swap(*mCluRefVecInp);
    if (mUseMC) {
      std::copy(mMCTruthInp->begin(), mMCTruthInp->end(), std::back_inserter(mMCTruthOut));
    }
  } else {
    for (int iev = 0; iev < n; iev++) {
      mTree->GetEntry(from + iev);
      //
      uint32_t shift = mCluRefVecOut.size(); // during accumulation clusters refs need to be shifted

      auto cl0 = mCluRefVecInp->begin();
      auto cl1 = mCluRefVecInp->end();
      std::copy(cl0, cl1, std::back_inserter(mCluRefVecOut));

      auto tr0 = mTracksInp->begin();
      auto tr1 = mTracksInp->end();
      // fix cluster references
      if (shift) {
        for (auto tr = tr0; tr != tr1; tr++) {
          tr->shiftFirstClusterRef(shift);
        }
      }
      std::copy(tr0, tr1, std::back_inserter(mTracksOut));
      // MC
      if (mUseMC) {
        std::copy(mMCTruthInp->begin(), mMCTruthInp->end(), std::back_inserter(mMCTruthOut));
      }
    }
  }
  LOG(INFO) << "TPCTrackReader pushes " << mTracksOut.size() << " tracks from entries " << from << " : " << from + n - 1;
}

void TrackReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  if (!(mFile && !mFile->IsZombie())) {
    throw std::runtime_error("Error opening tree file");
  }
  mTree.reset((TTree*)mFile->Get(mTrackTreeName.c_str()));
  if (!mTree) {
    throw std::runtime_error("Error opening tree");
  }

  mTree->SetBranchAddress(mTrackBranchName.c_str(), &mTracksInp);
  mTree->SetBranchAddress(mClusRefBranchName.c_str(), &mCluRefVecInp);
  if (mUseMC) {
    if (mTree->GetBranch(mTrackMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mTrackMCTruthBranchName.c_str(), &mMCTruthInp);
      LOG(INFO) << "Will use MC-truth from " << mTrackMCTruthBranchName;
    } else {
      LOG(INFO) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getTPCTrackReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("TPC", "TRACKS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("TPC", "CLUSREFS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("TPC", "TRACKSMCLBL", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tpc-track-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<TrackReader>(useMC)},
    Options{
      {"infile", VariantType::String, "tpctracks.root", {"Name of the input track file"}}}};
}

} // namespace tpc
} // namespace o2

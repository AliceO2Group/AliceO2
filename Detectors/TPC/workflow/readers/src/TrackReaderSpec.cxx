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
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCReaderWorkflow/TrackReaderSpec.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsGlobalTracking/TrackTuneParams.h"

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
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("infile"));
  mSkipClusRefs = ic.options().get<bool>("skip-clusref");
  connectTree(mInputFileName);
}

void TrackReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  accumulate(ent, 1);                // to really accumulate all, use accumulate(ent,mTree->GetEntries());
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  using TrackTunePar = o2::globaltracking::TrackTuneParams;
  const auto& trackTune = TrackTunePar::Instance();
  // Normally we should not apply tuning here as with sourceLevelTPC==true it is already applied in the tracking.
  // Note that there is no way to apply lumi scaling here!!!
  if ((trackTune.sourceLevelTPC && trackTune.applyWhenReading) &&
      (trackTune.useTPCInnerCorr || trackTune.useTPCOuterCorr ||
       trackTune.tpcCovInnerType != TrackTunePar::AddCovType::Disable || trackTune.tpcCovOuterType != TrackTunePar::AddCovType::Disable)) {
    for (auto& trc : mTracksOut) {
      if (trc.getNClusters() == 0) {
        continue; // filtered/reduced track
      }
      if (trackTune.useTPCInnerCorr) {
        trc.updateParams(trackTune.tpcParInner);
      }
      if (trackTune.tpcCovInnerType != TrackTunePar::AddCovType::Disable) {
        trc.updateCov(trackTune.tpcCovInner, trackTune.tpcCovInnerType == TrackTunePar::AddCovType::WithCorrelations);
      }
      if (trackTune.useTPCOuterCorr) {
        trc.getParamOut().updateParams(trackTune.tpcParOuter);
      }
      if (trackTune.tpcCovOuterType != TrackTunePar::AddCovType::Disable) {
        trc.getParamOut().updateCov(trackTune.tpcCovOuter, trackTune.tpcCovOuterType == TrackTunePar::AddCovType::WithCorrelations);
      }
    }
  }

  pc.outputs().snapshot(Output{"TPC", "TRACKS", 0}, mTracksOut);
  pc.outputs().snapshot(Output{"TPC", "CLUSREFS", 0}, mCluRefVecOut);
  if (mUseMC) {
    pc.outputs().snapshot(Output{"TPC", "TRACKSMCLBL", 0}, mMCTruthOut);
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
  LOG(info) << "TPCTrackReader pushes " << mTracksOut.size() << " tracks from entries " << from << " : " << from + n - 1;
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
  if (!mSkipClusRefs) {
    mTree->SetBranchAddress(mClusRefBranchName.c_str(), &mCluRefVecInp);
  } else {
    mCluRefVecInp = new std::vector<o2::tpc::TPCClRefElem>;
  }
  if (mUseMC) {
    if (mTree->GetBranch(mTrackMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mTrackMCTruthBranchName.c_str(), &mMCTruthInp);
      LOG(info) << "Will use MC-truth from " << mTrackMCTruthBranchName;
    } else {
      LOG(info) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
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
      {"infile", VariantType::String, "tpctracks.root", {"Name of the input track file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"skip-clusref", VariantType::Bool, false, {"Skip reading cluster references"}}}};
}

} // namespace tpc
} // namespace o2

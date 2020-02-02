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
}

void TrackReader::run(ProcessingContext& pc)
{

  if (mFinished) {
    return;
  }
  accumulate();

  LOG(INFO) << "ITSTrackReader pushes " << mROFRec.size() << " ROFRecords,"
            << mTracks.size() << " tracks";
  pc.outputs().snapshot(Output{mOrigin, "ITSTrackROF", 0, Lifetime::Timeframe}, mROFRec);
  pc.outputs().snapshot(Output{mOrigin, "TRACKS", 0, Lifetime::Timeframe}, mTracks);
  pc.outputs().snapshot(Output{mOrigin, "TRACKCLSID", 0, Lifetime::Timeframe}, mClusInd);
  pc.outputs().snapshot(Output{"ITS", "VERTICES", 0, Lifetime::Timeframe}, mVertices);
  pc.outputs().snapshot(Output{"ITS", "VERTICESROF", 0, Lifetime::Timeframe}, mVerticesROFRec);
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "TRACKSMCTR", 0, Lifetime::Timeframe}, mMCTruth);
  }

  mFinished = true;

  pc.services().get<ControlService>().endOfStream();
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
  if (!trTree->GetBranch(mROFBranchName.c_str())) {
    LOG(FATAL) << "Failed to load tracks ROF branch " << mROFBranchName << " from tree " << mTrackTreeName;
  }

  trTree->SetBranchAddress(mROFBranchName.c_str(), &mROFRecInp);
  trTree->SetBranchAddress(mTrackBranchName.c_str(), &mTracksInp);
  trTree->SetBranchAddress(mClusIdxBranchName.c_str(), &mClusIndInp);
  if (!trTree->GetBranch(mVertexBranchName.c_str())) {
    LOG(WARNING) << "No " << mVertexBranchName << " branch in " << mTrackTreeName << " -> vertices will be empty";
  } else {
    trTree->SetBranchAddress(mVertexBranchName.c_str(), &mVerticesInp);
  }
  if (!trTree->GetBranch(mVertexROFBranchName.c_str())) {
    LOG(WARNING) << "No " << mVertexROFBranchName << " branch in " << mTrackTreeName
                 << " -> vertices ROFrecords will be empty";
  } else {
    trTree->SetBranchAddress(mVertexROFBranchName.c_str(), &mVerticesROFRecInp);
  }

  if (mUseMC) {
    if (trTree->GetBranch(mTrackMCTruthBranchName.c_str())) {
      trTree->SetBranchAddress(mTrackMCTruthBranchName.c_str(), &mMCTruthInp);
      LOG(INFO) << "Will use MC-truth from " << mTrackMCTruthBranchName;
    } else {
      LOG(INFO) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  trTree->GetEntry(0);
  delete trTree;
  trFile.Close();
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

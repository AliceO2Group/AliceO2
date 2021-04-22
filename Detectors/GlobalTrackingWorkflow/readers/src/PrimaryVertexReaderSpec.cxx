// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   VertexReaderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GlobalTrackingWorkflowReaders/PrimaryVertexReaderSpec.h"
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

void PrimaryVertexReader::init(InitContext& ic)
{
  mFileName = o2::utils::concat_string(o2::base::NameConf::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                       ic.options().get<std::string>("primary-vertex-infile"));
  connectTree();
}

void PrimaryVertexReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "Pushing " << mVerticesPtr->size() << " vertices at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "PVTX", 0, Lifetime::Timeframe}, mVertices);
  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe}, mPV2MatchIdx);
  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe}, mPV2MatchIdxRef);

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "PVTX_MCTR", 0, Lifetime::Timeframe}, mLabels);
  }

  if (mVerbose) {
    int cnt = 0;
    for (const auto& vtx : mVertices) {
      Label lb;
      if (mUseMC) {
        lb = mLabels[cnt];
      }
      LOG(INFO) << "#" << cnt << " " << vtx << " | MC:" << lb;
      LOG(INFO) << "References: " << mPV2MatchIdxRef[cnt];
      for (int is = 0; is < GIndex::NSources; is++) {
        LOG(INFO) << GIndex::getSourceName(is) << " : " << mPV2MatchIdxRef[cnt].getEntriesOfSource(is) << " attached:";
        int idMin = mPV2MatchIdxRef[cnt].getFirstEntryOfSource(is), idMax = idMin + mPV2MatchIdxRef[cnt].getEntriesOfSource(is);
        std::string trIDs;
        int cntT = 0;
        for (int i = idMin; i < idMax; i++) {
          trIDs += mPV2MatchIdx[i].asString() + " ";
          if (!((++cntT) % 15)) {
            LOG(INFO) << trIDs;
            trIDs = "";
          }
        }
        if (!trIDs.empty()) {
          LOG(INFO) << trIDs;
        }
      }
      cnt++;
    }
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void PrimaryVertexReader::connectTree()
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mFileName.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mVertexTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mVertexBranchName.c_str()));
  assert(mTree->GetBranch(mVertexTrackIDsBranchName.c_str()));
  assert(mTree->GetBranch(mVertex2TrackIDRefsBranchName.c_str()));

  mTree->SetBranchAddress(mVertexBranchName.c_str(), &mVerticesPtr);
  mTree->SetBranchAddress(mVertexTrackIDsBranchName.c_str(), &mPV2MatchIdxPtr);
  mTree->SetBranchAddress(mVertex2TrackIDRefsBranchName.c_str(), &mPV2MatchIdxRefPtr);

  if (mUseMC) {
    assert(mTree->GetBranch(mVertexLabelsBranchName.c_str()));
    mTree->SetBranchAddress(mVertexLabelsBranchName.c_str(), &mLabelsPtr);
  }

  LOG(INFO) << "Loaded " << mVertexTreeName << " tree from " << mFileName << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getPrimaryVertexReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "PVTX", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe);

  if (useMC) {
    outputs.emplace_back("GLO", "PVTX_MCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "primary-vertex-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<PrimaryVertexReader>(useMC)},
    Options{
      {"primary-vertex-infile", VariantType::String, "o2_primary_vertex.root", {"Name of the input primary vertex file"}},
      {"vertex-track-matches-infile", VariantType::String, "o2_pvertex_track_matches.root", {"Name of the input file with primary vertex - tracks matches"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace vertexing
} // namespace o2

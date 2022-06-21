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

/// @file   VertexReaderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GlobalTrackingWorkflowReaders/PrimaryVertexReaderSpec.h"
#include "CommonUtils/NameConf.h"
#include "TFile.h"
#include "TTree.h"
#include "CommonDataFormat/TimeStamp.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "ReconstructionDataFormats/VtxTrackRef.h"
#include "ReconstructionDataFormats/PrimaryVertex.h"
#include "SimulationDataFormat/MCEventLabel.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

// read primary vertices produces by the o2-primary-vertexing-workflow
class PrimaryVertexReader : public o2::framework::Task
{
  using Label = o2::MCEventLabel;
  using V2TRef = o2::dataformats::VtxTrackRef;
  using PVertex = o2::dataformats::PrimaryVertex;
  using GIndex = o2::dataformats::VtxTrackIndex;

 public:
  PrimaryVertexReader(bool useMC) : mUseMC(useMC) {}
  ~PrimaryVertexReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree();

  bool mVerbose = false;
  bool mUseMC = false;

  std::vector<PVertex> mVertices, *mVerticesPtr = &mVertices;
  std::vector<Label> mLabels, *mLabelsPtr = &mLabels;
  std::vector<V2TRef> mPV2MatchIdxRef, *mPV2MatchIdxRefPtr = &mPV2MatchIdxRef;
  std::vector<GIndex> mPV2MatchIdx, *mPV2MatchIdxPtr = &mPV2MatchIdx;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::string mFileNameMatches = "";
  std::string mVertexTreeName = "o2sim";
  std::string mVertexBranchName = "PrimaryVertex";
  std::string mVertexTrackIDsBranchName = "PVTrackIndices";
  std::string mVertex2TrackIDRefsBranchName = "PV2TrackRefs";
  std::string mVertexLabelsBranchName = "PVMCTruth";
};

void PrimaryVertexReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("primary-vertex-infile"));
  mVerbose = ic.options().get<bool>("verbose-reader");
  connectTree();
}

void PrimaryVertexReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mVerticesPtr->size() << " vertices at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "PVTX", 0, Lifetime::Timeframe}, mVertices);
  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTC", 0, Lifetime::Timeframe}, mPV2MatchIdx);
  pc.outputs().snapshot(Output{"GLO", "PVTX_TRMTCREFS", 0, Lifetime::Timeframe}, mPV2MatchIdxRef);

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "PVTX_MCTR", 0, Lifetime::Timeframe}, mLabels);
  }

  if (mVerbose) {
    size_t nrec = mPV2MatchIdxRef.size();
    for (size_t cnt = 0; cnt < nrec; cnt++) {
      if (cnt < mVertices.size()) {
        const auto& vtx = mVertices[cnt];
        Label lb;
        if (mUseMC) {
          lb = mLabels[cnt];
        }
        LOG(info) << "#" << cnt << " " << mVertices[cnt] << " | MC:" << lb;
      } else {
        LOG(info) << "#" << cnt << " this is not a vertex";
      }
      LOG(info) << "References: " << mPV2MatchIdxRef[cnt];
      for (int is = 0; is < GIndex::NSources; is++) {
        LOG(info) << GIndex::getSourceName(is) << " : " << mPV2MatchIdxRef[cnt].getEntriesOfSource(is) << " attached:";
        int idMin = mPV2MatchIdxRef[cnt].getFirstEntryOfSource(is), idMax = idMin + mPV2MatchIdxRef[cnt].getEntriesOfSource(is);
        std::string trIDs;
        int cntT = 0;
        for (int i = idMin; i < idMax; i++) {
          trIDs += mPV2MatchIdx[i].asString() + " ";
          if (!((++cntT) % 15)) {
            LOG(info) << trIDs;
            trIDs = "";
          }
        }
        if (!trIDs.empty()) {
          LOG(info) << trIDs;
        }
      }
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

  LOG(info) << "Loaded " << mVertexTreeName << " tree from " << mFileName << " with " << mTree->GetEntries() << " entries";
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
      {"input-dir", VariantType::String, "none", {"Input directory"}},
      {"verbose-reader", VariantType::Bool, false, {"Print vertex/tracks info"}}}};
}

} // namespace vertexing
} // namespace o2

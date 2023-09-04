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

/// @file   SecondaryVertexReaderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GlobalTrackingWorkflowReaders/SecondaryVertexReaderSpec.h"
#include "CommonUtils/NameConf.h"
#include "CommonDataFormat/RangeReference.h"
#include "ReconstructionDataFormats/V0.h"
#include "ReconstructionDataFormats/Cascade.h"
#include "ReconstructionDataFormats/Decay3Body.h"
#include "TFile.h"
#include "TTree.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{
// read secondary vertices produces by the o2-secondary-vertexing-workflow
class SecondaryVertexReader : public o2::framework::Task
{
  using RRef = o2::dataformats::RangeReference<int, int>;
  using V0Index = o2::dataformats::V0Index;
  using V0 = o2::dataformats::V0;
  using CascadeIndex = o2::dataformats::CascadeIndex;
  using Cascade = o2::dataformats::Cascade;
  using Decay3BodyIndex = o2::dataformats::Decay3BodyIndex;
  using Decay3Body = o2::dataformats::Decay3Body;

 public:
  SecondaryVertexReader() = default;
  ~SecondaryVertexReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree();

  bool mVerbose = false;

  std::vector<V0Index> mV0sIdx, *mV0sIdxPtr = &mV0sIdx;
  std::vector<V0> mV0s, *mV0sPtr = &mV0s;
  std::vector<RRef> mPV2V0Ref, *mPV2V0RefPtr = &mPV2V0Ref;
  std::vector<CascadeIndex> mCascsIdx, *mCascsIdxPtr = &mCascsIdx;
  std::vector<Cascade> mCascs, *mCascsPtr = &mCascs;
  std::vector<RRef> mPV2CascRef, *mPV2CascRefPtr = &mPV2CascRef;
  std::vector<Decay3BodyIndex> m3BodysIdx, *m3BodysIdxPtr = &m3BodysIdx;
  std::vector<Decay3Body> m3Bodys, *m3BodysPtr = &m3Bodys;
  std::vector<RRef> mPV23BodyRef, *mPV23BodyRefPtr = &mPV23BodyRef;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::string mFileNameMatches = "";
  std::string mSVertexTreeName = "o2sim";
  std::string mV0IdxBranchName = "V0sID";
  std::string mV0BranchName = "V0s";
  std::string mPVertex2V0RefBranchName = "PV2V0Refs";
  std::string mCascIdxBranchName = "CascadesID";
  std::string mCascBranchName = "Cascades";
  std::string mPVertex2CascRefBranchName = "PV2CascRefs";
  std::string m3BodyIdxBranchName = "Decays3BodyID";
  std::string m3BodyBranchName = "Decays3Body";
  std::string mPVertex23BodyRefBranchName = "PV23BodyRefs";
};

void SecondaryVertexReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("secondary-vertex-infile"));
  connectTree();
}

void SecondaryVertexReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mV0s.size() << " V0s and " << mCascs.size() << " cascades at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "V0S_IDX", 0, Lifetime::Timeframe}, mV0sIdx);
  pc.outputs().snapshot(Output{"GLO", "V0S", 0, Lifetime::Timeframe}, mV0s);
  pc.outputs().snapshot(Output{"GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe}, mPV2V0Ref);
  pc.outputs().snapshot(Output{"GLO", "CASCS_IDX", 0, Lifetime::Timeframe}, mCascsIdx);
  pc.outputs().snapshot(Output{"GLO", "CASCS", 0, Lifetime::Timeframe}, mCascs);
  pc.outputs().snapshot(Output{"GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe}, mPV2CascRef);
  pc.outputs().snapshot(Output{"GLO", "DECAYS3BODY_IDX", 0, Lifetime::Timeframe}, m3BodysIdx);
  pc.outputs().snapshot(Output{"GLO", "DECAYS3BODY", 0, Lifetime::Timeframe}, m3Bodys);
  pc.outputs().snapshot(Output{"GLO", "PVTX_3BODYREFS", 0, Lifetime::Timeframe}, mPV23BodyRef);

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void SecondaryVertexReader::connectTree()
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(mFileName.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mSVertexTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mV0IdxBranchName.c_str()));
  assert(mTree->GetBranch(mV0BranchName.c_str()));
  assert(mTree->GetBranch(mPVertex2V0RefBranchName.c_str()));
  assert(mTree->GetBranch(mCascBranchName.c_str()));
  assert(mTree->GetBranch(mCascIdxBranchName.c_str()));
  assert(mTree->GetBranch(mPVertex2CascRefBranchName.c_str()));
  assert(mTree->GetBranch(m3BodyIdxBranchName.c_str()));
  assert(mTree->GetBranch(m3BodyBranchName.c_str()));
  assert(mTree->GetBranch(mPVertex23BodyRefBranchName.c_str()));

  mTree->SetBranchAddress(mV0IdxBranchName.c_str(), &mV0sIdxPtr);
  mTree->SetBranchAddress(mV0BranchName.c_str(), &mV0sPtr);
  mTree->SetBranchAddress(mPVertex2V0RefBranchName.c_str(), &mPV2V0RefPtr);
  mTree->SetBranchAddress(mCascIdxBranchName.c_str(), &mCascsIdxPtr);
  mTree->SetBranchAddress(mCascBranchName.c_str(), &mCascsPtr);
  mTree->SetBranchAddress(mPVertex2CascRefBranchName.c_str(), &mPV2CascRefPtr);
  mTree->SetBranchAddress(m3BodyIdxBranchName.c_str(), &m3BodysIdxPtr);
  mTree->SetBranchAddress(m3BodyBranchName.c_str(), &m3BodysPtr);
  mTree->SetBranchAddress(mPVertex23BodyRefBranchName.c_str(), &mPV23BodyRefPtr);

  LOG(info) << "Loaded " << mSVertexTreeName << " tree from " << mFileName << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getSecondaryVertexReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "V0S_IDX", 0, Lifetime::Timeframe);       // found V0s indices
  outputs.emplace_back("GLO", "V0S", 0, Lifetime::Timeframe);           // found V0s
  outputs.emplace_back("GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);   // prim.vertex -> V0s refs
  outputs.emplace_back("GLO", "CASCS_IDX", 0, Lifetime::Timeframe);     // found Cascades indices
  outputs.emplace_back("GLO", "CASCS", 0, Lifetime::Timeframe);         // found Cascades
  outputs.emplace_back("GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe); // prim.vertex -> Cascades refs
  outputs.emplace_back("GLO", "DECAYS3BODY_IDX", 0, Lifetime::Timeframe); // found 3 body vertices indices
  outputs.emplace_back("GLO", "DECAYS3BODY", 0, Lifetime::Timeframe);   // found 3 body Decays
  outputs.emplace_back("GLO", "PVTX_3BODYREFS", 0, Lifetime::Timeframe); // prim.vertex -> 3 body Decays refs

  return DataProcessorSpec{
    "secondary-vertex-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<SecondaryVertexReader>()},
    Options{
      {"secondary-vertex-infile", VariantType::String, "o2_secondary_vertex.root", {"Name of the input secondary vertex file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace vertexing
} // namespace o2

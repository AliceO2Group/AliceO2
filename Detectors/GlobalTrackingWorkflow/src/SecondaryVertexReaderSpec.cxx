// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SecondaryVertexReaderSpec.cxx

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GlobalTrackingWorkflow/SecondaryVertexReaderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

void SecondaryVertexReader::init(InitContext& ic)
{
  mFileName = ic.options().get<std::string>("secondary-vertex-infile");
  connectTree();
}

void SecondaryVertexReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "Pushing " << mV0s.size() << " V0s and " << mCascs.size() << " cascades at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "V0S", 0, Lifetime::Timeframe}, mV0s);
  pc.outputs().snapshot(Output{"GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe}, mPV2V0Ref);
  pc.outputs().snapshot(Output{"GLO", "CASCS", 0, Lifetime::Timeframe}, mCascs);
  pc.outputs().snapshot(Output{"GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe}, mPV2CascRef);

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
  assert(mTree->GetBranch(mV0BranchName.c_str()));
  assert(mTree->GetBranch(mPVertex2V0RefBranchName.c_str()));
  assert(mTree->GetBranch(mCascBranchName.c_str()));
  assert(mTree->GetBranch(mPVertex2CascRefBranchName.c_str()));

  mTree->SetBranchAddress(mV0BranchName.c_str(), &mV0sPtr);
  mTree->SetBranchAddress(mPVertex2V0RefBranchName.c_str(), &mPV2V0RefPtr);
  mTree->SetBranchAddress(mCascBranchName.c_str(), &mCascsPtr);
  mTree->SetBranchAddress(mPVertex2CascRefBranchName.c_str(), &mPV2CascRefPtr);

  LOG(INFO) << "Loaded " << mSVertexTreeName << " tree from " << mFileName << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getSecondaryVertexReaderSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "V0S", 0, Lifetime::Timeframe);           // found V0s
  outputs.emplace_back("GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);   // prim.vertex -> V0s refs
  outputs.emplace_back("GLO", "CASCS", 0, Lifetime::Timeframe);         // found Cascades
  outputs.emplace_back("GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe); // prim.vertex -> Cascades refs

  return DataProcessorSpec{
    "secondary-vertex-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<SecondaryVertexReader>()},
    Options{
      {"secondary-vertex-infile", VariantType::String, "o2_secondary_vertex.root", {"Name of the input secondary vertex file"}}}};
}

} // namespace vertexing
} // namespace o2

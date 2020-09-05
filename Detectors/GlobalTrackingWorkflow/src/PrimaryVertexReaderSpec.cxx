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
#include "GlobalTrackingWorkflow/PrimaryVertexReaderSpec.h"

using namespace o2::framework;

namespace o2
{
namespace vertexing
{

void PrimaryVertexReader::init(InitContext& ic)
{
  mFileName = ic.options().get<std::string>("primary-vertex-infile");
  connectTree(mFileName);
}

void PrimaryVertexReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << "Pushing " << mVerticesPtr->size() << " vertices at entry " << ent;

  pc.outputs().snapshot(Output{"GLO", "PVERTEX", 0, Lifetime::Timeframe}, mVertices);
  pc.outputs().snapshot(Output{"GLO", "PVERTEX_TRIDREFS", 0, Lifetime::Timeframe}, mPV2TrIdx);
  pc.outputs().snapshot(Output{"GLO", "PVERTEX_TRID", 0, Lifetime::Timeframe}, mPVTrIdx);

  if (mUseMC) {
    pc.outputs().snapshot(Output{"GLO", "PVERTEX_MCTR", 0, Lifetime::Timeframe}, mLabels);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void PrimaryVertexReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mVertexTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mVertexBranchName.c_str()));
  assert(mTree->GetBranch(mVertexTrackIDsBranchName.c_str()));
  assert(mTree->GetBranch(mVertex2TrackIDRefsBranchName.c_str()));

  mTree->SetBranchAddress(mVertexBranchName.c_str(), &mVerticesPtr);
  mTree->SetBranchAddress(mVertexTrackIDsBranchName.c_str(), &mPVTrIdxPtr);
  mTree->SetBranchAddress(mVertex2TrackIDRefsBranchName.c_str(), &mPV2TrIdxPtr);

  if (mUseMC) {
    assert(mTree->GetBranch(mVertexLabelsBranchName.c_str()));
    mTree->SetBranchAddress(mVertexLabelsBranchName.c_str(), &mLabelsPtr);
  }

  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getPrimaryVertexReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("GLO", "PVERTEX", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVERTEX_TRIDREFS", 0, Lifetime::Timeframe);
  outputs.emplace_back("GLO", "PVERTEX_TRID", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("GLO", "PVERTEX_MCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "primary-vertex-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<PrimaryVertexReader>(useMC)},
    Options{
      {"primary-vertex-infile", VariantType::String, "o2_primary_vertex.root", {"Name of the input primary vertex file"}}}};
}

} // namespace vertexing
} // namespace o2

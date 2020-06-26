// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterReaderSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "EndCapsWorkflow/ClusterReaderSpec.h"
#include <cassert>

using namespace o2::framework;
using namespace o2::endcaps;

namespace o2
{
namespace endcaps
{

ClusterReader::ClusterReader(o2::detectors::DetID id, bool useMC, bool useClFull, bool useClComp, bool usePatterns)
{
  assert(id == o2::detectors::DetID::EC0);
  mDetNameLC = mDetName = id.getName();
  mUseMC = useMC;
  mUseClFull = useClFull;
  mUseClComp = useClComp;
  mUsePatterns = usePatterns;
  std::transform(mDetNameLC.begin(), mDetNameLC.end(), mDetNameLC.begin(), ::tolower);
}

void ClusterReader::init(InitContext& ic)
{
  mFileName = ic.options().get<std::string>((mDetNameLC + "-cluster-infile").c_str());
  connectTree(mFileName);
}

void ClusterReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << mDetName << "ClusterReader pushes " << mClusROFRec.size() << " ROFRecords,"
            << mClusterArray.size() << " full clusters, " << mClusterCompArray.size()
            << " compact clusters at entry " << ent;

  // This is a very ugly way of providing DataDescription, which anyway does not need to contain detector name.
  // To be fixed once the names-definition class is ready
  pc.outputs().snapshot(Output{mOrigin, "CLUSTERSROF", 0, Lifetime::Timeframe}, mClusROFRec);
  if (mUseClFull) {
    pc.outputs().snapshot(Output{mOrigin, "CLUSTERS", 0, Lifetime::Timeframe}, mClusterArray);
  }
  if (mUseClComp) {
    pc.outputs().snapshot(Output{mOrigin, "COMPCLUSTERS", 0, Lifetime::Timeframe}, mClusterCompArray);
  }
  if (mUsePatterns) {
    pc.outputs().snapshot(Output{mOrigin, "PATTERNS", 0, Lifetime::Timeframe}, mPatternsArray);
  }
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, mClusterMCTruth);
    pc.outputs().snapshot(Output{mOrigin, "CLUSTERSMC2ROF", 0, Lifetime::Timeframe}, mClusMC2ROFs);
  }

  if (mTree->GetReadEntry() + 1 >= mTree->GetEntries()) {
    pc.services().get<ControlService>().endOfStream();
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
  }
}

void ClusterReader::connectTree(const std::string& filename)
{
  mTree.reset(nullptr); // in case it was already loaded
  mFile.reset(TFile::Open(filename.c_str()));
  assert(mFile && !mFile->IsZombie());
  mTree.reset((TTree*)mFile->Get(mClusTreeName.c_str()));
  assert(mTree);

  mTree->SetBranchAddress((mDetName + mClusROFBranchName).c_str(), &mClusROFRecPtr);
  if (mUseClFull) {
    mTree->SetBranchAddress((mDetName + mClusterBranchName).c_str(), &mClusterArrayPtr); // being eliminated!!!
  }
  if (mUseClComp) {
    mTree->SetBranchAddress((mDetName + mClusterCompBranchName).c_str(), &mClusterCompArrayPtr);
  }
  if (mUsePatterns) {
    mTree->SetBranchAddress((mDetName + mClusterPattBranchName).c_str(), &mPatternsArrayPtr);
  }
  if (mUseMC) {
    if (mTree->GetBranch((mDetName + mClustMCTruthBranchName).c_str()) &&
        mTree->GetBranch((mDetName + mClustMC2ROFBranchName).c_str())) {
      mTree->SetBranchAddress((mDetName + mClustMCTruthBranchName).c_str(), &mClusterMCTruthPtr);
      mTree->SetBranchAddress((mDetName + mClustMC2ROFBranchName).c_str(), &mClusMC2ROFsPtr);
    } else {
      LOG(INFO) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  LOG(INFO) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getEC0ClusterReaderSpec(bool useMC, bool useClFull, bool useClComp, bool usePatterns)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("EC0", "CLUSTERSROF", 0, Lifetime::Timeframe);
  if (useClFull) {
    outputSpec.emplace_back("EC0", "CLUSTERS", 0, Lifetime::Timeframe);
  }
  if (useClComp) {
    outputSpec.emplace_back("EC0", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  }
  if (usePatterns) {
    outputSpec.emplace_back("EC0", "PATTERNS", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    outputSpec.emplace_back("EC0", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputSpec.emplace_back("EC0", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "ec0-cluster-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<EC0ClusterReader>(useMC, useClFull, useClComp)},
    Options{
      {"ec0-cluster-infile", VariantType::String, "o2clus_ec0.root", {"Name of the input cluster file"}}}};
}

} // namespace endcaps
} // namespace o2

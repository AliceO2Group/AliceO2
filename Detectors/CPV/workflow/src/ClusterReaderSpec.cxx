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

/// @file   ClusterReaderSpec.cxx

#include <vector>
#include <cassert>
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "CPVWorkflow/ClusterReaderSpec.h"
#include "CommonUtils/NameConf.h"

using namespace o2::framework;
using namespace o2::cpv;

namespace o2
{
namespace cpv
{

ClusterReader::ClusterReader(bool useMC)
{
  mUseMC = useMC;
}

void ClusterReader::init(InitContext& ic)
{
  mInputFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                                 ic.options().get<std::string>("cpv-clusters-infile"));
  connectTree(mInputFileName);
}

void ClusterReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(info) << "Pushing " << mClusters.size() << " Clusters in " << mTRs.size() << " TriggerRecords at entry " << ent;
  pc.outputs().snapshot(Output{mOrigin, "CLUSTERS", 0}, mClusters);
  pc.outputs().snapshot(Output{mOrigin, "CLUSTERTRIGRECS", 0}, mTRs);
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "CLUSTERTRUEMC", 0}, mMCTruth);
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
  mTree.reset((TTree*)mFile->Get(mClusterTreeName.c_str()));
  assert(mTree);
  assert(mTree->GetBranch(mTRBranchName.c_str()));

  mTree->SetBranchAddress(mTRBranchName.c_str(), &mTRsInp);
  mTree->SetBranchAddress(mClusterBranchName.c_str(), &mClustersInp);
  if (mUseMC) {
    if (mTree->GetBranch(mClusterMCTruthBranchName.c_str())) {
      mTree->SetBranchAddress(mClusterMCTruthBranchName.c_str(), &mMCTruthInp);
    } else {
      LOG(warning) << "MC-truth is missing, message will be empty";
    }
  }
  LOG(info) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getCPVClusterReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("CPV", "CLUSTERS", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("CPV", "CLUSTERTRIGRECS", 0, Lifetime::Timeframe);
  if (useMC) {
    outputSpec.emplace_back("CPV", "CLUSTERTRUEMC", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "cpv-cluster-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<ClusterReader>(useMC)},
    Options{
      {"cpv-clusters-infile", VariantType::String, "cpvclusters.root", {"Name of the input Cluster file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace cpv
} // namespace o2

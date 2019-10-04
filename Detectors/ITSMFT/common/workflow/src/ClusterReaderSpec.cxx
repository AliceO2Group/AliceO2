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
#include "ITSMFTWorkflow/ClusterReaderSpec.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace itsmft
{

ClusterReader::ClusterReader(o2::detectors::DetID id, bool useMC, bool useClFull, bool useClComp)
{
  assert(id == o2::detectors::DetID::ITS || id == o2::detectors::DetID::MFT);
  mDetNameLC = mDetName = id.getName();
  mUseMC = useMC;
  mUseClFull = useClFull;
  mUseClComp = useClComp;
  std::transform(mDetNameLC.begin(), mDetNameLC.end(), mDetNameLC.begin(), ::tolower);
}

void ClusterReader::init(InitContext& ic)
{
  mInputFileName = ic.options().get<std::string>((mDetNameLC + "-cluster-infile").c_str());
}

void ClusterReader::run(ProcessingContext& pc)
{

  if (mFinished) {
    return;
  }
  accumulate();

  LOG(INFO) << mDetName << "ClusterReader pushes " << mClusROFRecOut.size() << " ROFRecords,"
            << mClusterArrayOut.size() << " full clusters, " << mClusterCompArrayOut.size()
            << " compact clusters";

  // This is a very ugly way of providing DataDescription, which anyway does not need to contain detector name.
  // To be fixed once the names-definition class is ready
  pc.outputs().snapshot(Output{mOrigin, mOrigin == o2::header::gDataOriginITS ? "ITSClusterROF" : "MFTClusterROF",
                               0, Lifetime::Timeframe},
                        mClusROFRecOut);
  if (mUseClFull) {
    pc.outputs().snapshot(Output{mOrigin, "CLUSTERS", 0, Lifetime::Timeframe}, mClusterArrayOut);
  }
  if (mUseClComp) {
    pc.outputs().snapshot(Output{mOrigin, "COMPCLUSTERS", 0, Lifetime::Timeframe}, mClusterCompArrayOut);
  }
  if (mUseMC) {
    pc.outputs().snapshot(Output{mOrigin, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, mClusterMCTruthOut);
  }

  mFinished = true;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

void ClusterReader::accumulate()
{
  // load data from files
  TFile clFile(mInputFileName.c_str(), "read");
  if (clFile.IsZombie()) {
    LOG(FATAL) << "Failed to open cluster file " << mInputFileName;
  }
  TTree* clTree = (TTree*)clFile.Get(mClusTreeName.c_str());
  if (!clTree) {
    LOG(FATAL) << "Failed to load clusters tree " << mClusTreeName << " from " << mInputFileName;
  }
  TTree* rofTree = (TTree*)clFile.Get((mDetName + mClusROFTreeName).c_str());
  if (!rofTree) {
    LOG(FATAL) << "Failed to load clusters ROF tree " << rofTree << " from " << mInputFileName;
  }
  LOG(INFO) << "Loaded cluter tree " << mClusTreeName << " and ROFRecords " << mClusROFTreeName << " from " << mInputFileName;

  rofTree->SetBranchAddress((mDetName + mClusROFTreeName).c_str(), &mClusROFRecInp);

  if (mUseClFull) {
    clTree->SetBranchAddress((mDetName + mClusterBranchName).c_str(), &mClusterArrayInp);
  }
  if (mUseClComp) {
    clTree->SetBranchAddress((mDetName + mClusterCompBranchName).c_str(), &mClusterCompArrayInp);
  }
  if (mUseMC) {
    if (clTree->GetBranch((mDetName + mClustMCTruthBranchName).c_str())) {
      clTree->SetBranchAddress((mDetName + mClustMCTruthBranchName).c_str(), &mClusterMCTruthInp);
      LOG(INFO) << "Will use MC-truth from " << mDetName + mClustMCTruthBranchName;
    } else {
      LOG(INFO) << "MC-truth is missing";
      mUseMC = false;
    }
  }
  // it is possible that the cluster data is stored in multiple entries, in this case we need to refill to 1 single vector
  if (rofTree->GetEntries() > 1) {
    LOG(FATAL) << "Clusters ROFRecords tree has " << rofTree->GetEntries() << " entries instead of 1";
  }
  rofTree->GetEntry(0);
  int nROFs = mClusROFRecInp->size();
  mClusROFRecOut.swap(*mClusROFRecInp);
  int nEnt = clTree->GetEntries();
  if (nEnt == 1) {
    clTree->GetEntry(0);
    if (mUseClFull) {
      mClusterArrayOut.swap(*mClusterArrayInp);
    }
    if (mUseClComp) {
      mClusterCompArrayOut.swap(*mClusterCompArrayInp);
    }
    if (mUseMC) {
      mClusterMCTruthOut.mergeAtBack(*mClusterMCTruthInp);
    }
  } else {
    int lastEntry = -1;
    int nclAcc = 0;
    for (auto& rof : mClusROFRecOut) {
      auto rEntry = rof.getROFEntry().getEvent();
      if (lastEntry != rEntry) {
        clTree->GetEntry((lastEntry = rEntry));
      }
      // full clusters
      if (mUseClFull) {
        auto cl0 = mClusterArrayInp->begin() + rof.getROFEntry().getIndex();
        auto cl1 = cl0 + rof.getNROFEntries();
        std::copy(cl0, cl1, std::back_inserter(mClusterArrayOut));
      }
      // compact clusters
      if (mUseClComp) {
        auto cl0 = mClusterCompArrayInp->begin() + rof.getROFEntry().getIndex();
        auto cl1 = cl0 + rof.getNROFEntries();
        std::copy(cl0, cl1, std::back_inserter(mClusterCompArrayOut));
      }
      // MC
      if (mUseMC) {
        mClusterMCTruthOut.mergeAtBack(*mClusterMCTruthInp);
      }
      rof.getROFEntry().setEvent(0);
      rof.getROFEntry().setIndex(nclAcc);
      nclAcc += rof.getNROFEntries();
    }
  }
}

DataProcessorSpec getITSClusterReaderSpec(bool useMC, bool useClFull, bool useClComp)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("ITS", "ITSClusterROF", 0, Lifetime::Timeframe);
  if (useClFull) {
    outputSpec.emplace_back("ITS", "CLUSTERS", 0, Lifetime::Timeframe);
  }
  if (useClComp) {
    outputSpec.emplace_back("ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    outputSpec.emplace_back("ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-cluster-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<ITSClusterReader>(useMC, useClFull, useClComp)},
    Options{
      {"its-cluster-infile", VariantType::String, "o2clus_its.root", {"Name of the input cluster file"}}}};
}

DataProcessorSpec getMFTClusterReaderSpec(bool useMC, bool useClFull, bool useClComp)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("MFT", "MFTClusterROF", 0, Lifetime::Timeframe);
  if (useClFull) {
    outputSpec.emplace_back("MFT", "CLUSTERS", 0, Lifetime::Timeframe);
  }
  if (useClComp) {
    outputSpec.emplace_back("MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    outputSpec.emplace_back("MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-cluster-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<MFTClusterReader>(useMC, useClFull, useClComp)},
    Options{
      {"mft-cluster-infile", VariantType::String, "o2clus_mft.root", {"Name of the input cluster file"}}}};
}

} // namespace itsmft
} // namespace o2

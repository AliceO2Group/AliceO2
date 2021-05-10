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
#include "Framework/Logger.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include <cassert>
#include "DetectorsCommonDataFormats/NameConf.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace itsmft
{

ClusterReader::ClusterReader(o2::detectors::DetID id, bool useMC, bool usePatterns)
{
  assert(id == o2::detectors::DetID::ITS || id == o2::detectors::DetID::MFT);
  mDetNameLC = mDetName = id.getName();
  mUseMC = useMC;
  mUsePatterns = usePatterns;
  std::transform(mDetNameLC.begin(), mDetNameLC.end(), mDetNameLC.begin(), ::tolower);
}

void ClusterReader::init(InitContext& ic)
{
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>((mDetNameLC + "-cluster-infile").c_str()));
  connectTree(mFileName);
}

void ClusterReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(INFO) << mDetName << "ClusterReader pushes " << mClusROFRec.size() << " ROFRecords,"
            << mClusterCompArray.size() << " compact clusters at entry " << ent;

  // This is a very ugly way of providing DataDescription, which anyway does not need to contain detector name.
  // To be fixed once the names-definition class is ready
  pc.outputs().snapshot(Output{mOrigin, "CLUSTERSROF", 0, Lifetime::Timeframe}, mClusROFRec);
  pc.outputs().snapshot(Output{mOrigin, "COMPCLUSTERS", 0, Lifetime::Timeframe}, mClusterCompArray);
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
  mTree->SetBranchAddress((mDetName + mClusterCompBranchName).c_str(), &mClusterCompArrayPtr);
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

DataProcessorSpec getITSClusterReaderSpec(bool useMC, bool usePatterns)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  if (usePatterns) {
    outputSpec.emplace_back("ITS", "PATTERNS", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    outputSpec.emplace_back("ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputSpec.emplace_back("ITS", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-cluster-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<ITSClusterReader>(useMC, usePatterns)},
    Options{
      {"its-cluster-infile", VariantType::String, "o2clus_its.root", {"Name of the input cluster file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

DataProcessorSpec getMFTClusterReaderSpec(bool useMC, bool usePatterns)
{
  std::vector<OutputSpec> outputSpec;
  outputSpec.emplace_back("MFT", "CLUSTERSROF", 0, Lifetime::Timeframe);
  outputSpec.emplace_back("MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  if (usePatterns) {
    outputSpec.emplace_back("MFT", "PATTERNS", 0, Lifetime::Timeframe);
  }
  if (useMC) {
    outputSpec.emplace_back("MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputSpec.emplace_back("MFT", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-cluster-reader",
    Inputs{},
    outputSpec,
    AlgorithmSpec{adaptFromTask<MFTClusterReader>(useMC, usePatterns)},
    Options{
      {"mft-cluster-infile", VariantType::String, "o2clus_mft.root", {"Name of the input cluster file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace itsmft
} // namespace o2

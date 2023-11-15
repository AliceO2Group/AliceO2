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

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "TOFWorkflowIO/ClusterReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "TOFBase/Geo.h"

using namespace o2::framework;
using namespace o2::tof;

namespace o2
{
namespace tof
{

void ClusterReader::init(InitContext& ic)
{
  LOG(debug) << "Init Cluster reader!";
  mFileName = o2::utils::Str::concat_string(o2::utils::Str::rectifyDirectory(ic.options().get<std::string>("input-dir")),
                                            ic.options().get<std::string>("tof-cluster-infile"));
  connectTree(mFileName);
}

void ClusterReader::run(ProcessingContext& pc)
{
  auto ent = mTree->GetReadEntry() + 1;
  assert(ent < mTree->GetEntries()); // this should not happen
  mTree->GetEntry(ent);
  LOG(debug) << "Pushing " << mClustersPtr->size() << " TOF clusters at entry " << ent;

  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe}, mClusters);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERSMULT", 0, Lifetime::Timeframe}, mClustersMult);
  if (mUseMC) {
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe}, mLabels);
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
  mTree.reset((TTree*)mFile->Get("o2sim"));
  assert(mTree);
  mTree->SetBranchAddress("TOFCluster", &mClustersPtr);

  if (mTree->GetBranch("TOFClusterMult")) {
    mTree->SetBranchAddress("TOFClusterMult", &mClustersMultPtr);
  } else {
    mClustersMult.resize(o2::base::GRPGeomHelper::instance().getNHBFPerTF() * o2::constants::lhc::LHCMaxBunches);
  }

  if (mUseMC) {
    mTree->SetBranchAddress("TOFClusterMCTruth", &mLabelsPtr);
  }
  LOG(debug) << "Loaded tree from " << filename << " with " << mTree->GetEntries() << " entries";
}

DataProcessorSpec getClusterReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(o2::header::gDataOriginTOF, "CLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back(o2::header::gDataOriginTOF, "CLUSTERSMULT", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back(o2::header::gDataOriginTOF, "CLUSTERSMCTR", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "tof-cluster-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<ClusterReader>(useMC)},
    Options{
      {"tof-cluster-infile", VariantType::String, "tofclusters.root", {"Name of the input file"}},
      {"input-dir", VariantType::String, "none", {"Input directory"}}}};
}

} // namespace tof
} // namespace o2

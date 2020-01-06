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

#include "MFTWorkflow/ClusterReaderSpec.h"

#include "TTree.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;
using namespace o2::itsmft;

namespace o2
{
namespace mft
{

void ClusterReader::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("mft-cluster-infile");
  mFile = std::make_unique<TFile>(filename.c_str(), "OLD");
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    mState = 0;
    return;
  }
  mState = 1;
}

void ClusterReader::run(ProcessingContext& pc)
{
  if (mState != 1)
    return;

  std::unique_ptr<TTree> tree((TTree*)mFile->Get("o2sim"));
  std::unique_ptr<std::vector<ROFRecord>> rofs((std::vector<ROFRecord>*)mFile->Get("MFTClusterROF"));

  if (tree && rofs) {

    std::vector<o2::itsmft::Cluster> allClusters;
    std::vector<o2::itsmft::CompClusterExt> allCompClusters;
    std::vector<o2::itsmft::Cluster> clusters, *pclusters = &clusters;
    std::vector<o2::itsmft::CompClusterExt> compClusters, *pcompClusters = &compClusters;
    tree->SetBranchAddress("MFTCluster", &pclusters);
    tree->SetBranchAddress("MFTClusterComp", &pcompClusters);

    std::unique_ptr<TTree> treeMC2ROF;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> allLabels;
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> labels, *plabels = &labels;
    std::vector<MC2ROFRecord> mc2rofs, *pmc2rofs = &mc2rofs;
    if (mUseMC) {
      tree->SetBranchAddress("MFTClusterMCTruth", &plabels);
      treeMC2ROF.reset((TTree*)mFile->Get("MFTClusterMC2ROF"));
      if (treeMC2ROF) {
        treeMC2ROF->SetBranchAddress("MFTDigitMC2ROF", &pmc2rofs);
        treeMC2ROF->GetEntry(0);
      }
    }

    int ne = tree->GetEntries();
    for (int e = 0; e < ne; e++) {
      tree->GetEntry(e);
      std::copy(clusters.begin(), clusters.end(), std::back_inserter(allClusters));
      std::copy(compClusters.begin(), compClusters.end(), std::back_inserter(allCompClusters));
      allLabels.mergeAtBack(labels);
    }
    LOG(INFO) << "MFTClusterReader pulled " << clusters.size() << " clusters, in "
              << rofs->size() << " RO frames";

    pc.outputs().snapshot(Output{"MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe}, allCompClusters);
    pc.outputs().snapshot(Output{"MFT", "CLUSTERS", 0, Lifetime::Timeframe}, allClusters);
    pc.outputs().snapshot(Output{"MFT", "MFTClusterROF", 0, Lifetime::Timeframe}, *rofs);
    if (mUseMC) {
      pc.outputs().snapshot(Output{"MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe}, allLabels);
      pc.outputs().snapshot(Output{"MFT", "MFTClusterMC2ROF", 0, Lifetime::Timeframe}, mc2rofs);
    }
  } else {
    LOG(ERROR) << "Cannot read the MFT clusters !";
    return;
  }
  mState = 2;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getClusterReaderSpec(bool useMC)
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "CLUSTERS", 0, Lifetime::Timeframe);
  outputs.emplace_back("MFT", "MFTClusterROF", 0, Lifetime::Timeframe);
  if (useMC) {
    outputs.emplace_back("MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("MFT", "MFTClusterMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "mft-cluster-reader",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<ClusterReader>(useMC)},
    Options{
      {"mft-cluster-infile", VariantType::String, "mftclusters.root", {"Name of the input file"}}}};
}

} // namespace mft
} // namespace o2

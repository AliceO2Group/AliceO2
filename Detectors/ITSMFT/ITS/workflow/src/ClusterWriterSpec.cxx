// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   ClusterWriterSpec.cxx

#include <vector>

#include "TTree.h"

#include "Framework/ControlService.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

void ClusterWriter::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("its-cluster-outfile");
  mFile = std::make_unique<TFile>(filename.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    LOG(ERROR) << "Cannot open the " << filename.c_str() << " file !";
    mState = 0;
    return;
  }
  mState = 1;
}

void ClusterWriter::run(ProcessingContext& pc)
{
  if (mState != 1)
    return;

  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  auto clusters = pc.inputs().get<const std::vector<o2::itsmft::Cluster>>("clusters");
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* plabels = nullptr;

  LOG(INFO) << "ITSClusterWriter pulled " << clusters.size() << " clusters, in "
            << rofs.size() << " RO frames";

  TTree tree("o2sim", "Tree with ITS clusters");
  tree.Branch("ITSClusterComp", &compClusters);
  tree.Branch("ITSCluster", &clusters);
  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    plabels = labels.get();
    tree.Branch("ITSClusterMCTruth", &plabels);
  }
  tree.Fill();
  tree.Write();

  // write ROFrecords vector to a tree
  TTree treeROF("ITSClustersROF", "ROF records tree");
  auto* rofsPtr = &rofs;
  treeROF.Branch("ITSClustersROF", &rofsPtr);
  treeROF.Fill();
  treeROF.Write();

  if (mUseMC) {
    // write MC2ROFrecord vector (directly inherited from digits input) to a tree
    TTree treeMC2ROF("ITSClustersMC2ROF", "MC -> ROF records tree");
    auto mc2rofs = pc.inputs().get<const std::vector<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
    auto* mc2rofsPtr = &mc2rofs;
    treeMC2ROF.Branch("ITSClustersMC2ROF", &mc2rofsPtr);
    treeMC2ROF.Fill();
    treeMC2ROF.Write();
  }

  mFile->Close();

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
}

DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("clusters", "ITS", "CLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "ITSClusterROF", 0, Lifetime::Timeframe);
  if (useMC) {
    inputs.emplace_back("labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("MC2ROframes", "ITS", "ITSClusterMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-cluster-writer",
    inputs,
    Outputs{},
    AlgorithmSpec{adaptFromTask<ClusterWriter>(useMC)},
    Options{
      {"its-cluster-outfile", VariantType::String, "o2clus_its.root", {"Name of the output file"}}}};
}

} // namespace its
} // namespace o2

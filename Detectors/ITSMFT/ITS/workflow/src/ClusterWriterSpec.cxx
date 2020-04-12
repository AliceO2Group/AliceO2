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

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

template <typename T>
TBranch* getOrMakeBranch(TTree* tree, const char* brname, T* ptr)
{
  if (auto br = tree->GetBranch(brname)) {
    br->SetAddress(static_cast<void*>(ptr));
    return br;
  }
  return tree->Branch(brname, ptr); // otherwise make it
}

void ClusterWriter::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("its-cluster-outfile");
  mFile = std::make_unique<TFile>(filename.c_str(), "RECREATE");
  if (!mFile->IsOpen()) {
    throw std::runtime_error(o2::utils::concat_string("failed to open ITS clusters output file ", filename));
  }
  mTree = std::make_unique<TTree>("o2sim", "Tree with ITS clusters");
}

void ClusterWriter::run(ProcessingContext& pc)
{
  auto compClusters = pc.inputs().get<const std::vector<o2::itsmft::CompClusterExt>>("compClusters");
  auto pspan = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto clusters = pc.inputs().get<const std::vector<o2::itsmft::Cluster>>("clusters");
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");

  std::unique_ptr<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>> labels;
  const o2::dataformats::MCTruthContainer<o2::MCCompLabel>* plabels = nullptr;
  std::vector<o2::itsmft::MC2ROFRecord> mc2rofs, *mc2rofsPtr = &mc2rofs;
  std::vector<unsigned char> patterns(pspan.begin(), pspan.end());

  LOG(INFO) << "ITSClusterWriter pulled " << compClusters.size() << " clusters, in " << rofs.size() << " RO frames";
  auto compClustersPtr = &compClusters;
  getOrMakeBranch(mTree.get(), "ITSClusterComp", &compClustersPtr);
  auto patternsPtr = &patterns;
  getOrMakeBranch(mTree.get(), "ITSClusterPatt", &patternsPtr);
  auto clustersPtr = &clusters;
  getOrMakeBranch(mTree.get(), "ITSCluster", &clustersPtr); // RSTODO being eliminated
  auto rofsPtr = &rofs;
  getOrMakeBranch(mTree.get(), "ITSClustersROF", &rofsPtr);

  if (mUseMC) {
    labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
    plabels = labels.get();
    getOrMakeBranch(mTree.get(), "ITSClusterMCTruth", &plabels);

    // write MC2ROFrecord vector (directly inherited from digits input) to a tree
    const auto m2rvec = pc.inputs().get<gsl::span<o2::itsmft::MC2ROFRecord>>("MC2ROframes");
    mc2rofs.reserve(m2rvec.size());
    for (const auto& m2rv : m2rvec) {
      mc2rofs.push_back(m2rv);
    }
    getOrMakeBranch(mTree.get(), "ITSClustersMC2ROF", &mc2rofsPtr);
  }
  mTree->Fill();
}

void ClusterWriter::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  LOG(INFO) << "Finalizing ITS cluster writing";
  mTree->Write();
  mTree.release()->Delete();
  mFile->Close();
}

DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
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

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
#include "MFTWorkflow/ClusterWriterSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

using namespace o2::framework;

namespace o2
{
namespace mft
{

void ClusterWriter::init(InitContext& ic)
{
  auto filename = ic.options().get<std::string>("mft-cluster-outfile");
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
  auto labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
  auto plabels = labels.get();
  auto rofs = pc.inputs().get<const std::vector<o2::itsmft::ROFRecord>>("ROframes");
  auto mc2rofs = pc.inputs().get<const std::vector<o2::itsmft::MC2ROFRecord>>("MC2ROframes");

  LOG(INFO) << "MFTClusterWriter pulled " << clusters.size() << " clusters, "
            << labels->getIndexedSize() << " MC label objects, in "
            << rofs.size() << " RO frames and "
            << mc2rofs.size() << " MC events";

  mFile->WriteObjectAny(&rofs, "std::vector<o2::itsmft::ROFRecord>", "MFTClusterROF");
  mFile->WriteObjectAny(&mc2rofs, "std::vector<o2::itsmft::MC2ROFRecord>", "MFTClusterMC2ROF");

  TTree tree("o2sim", "Tree with MFT clusters");
  tree.Branch("MFTClusterComp", &compClusters);
  tree.Branch("MFTCluster", &clusters);
  tree.Branch("MFTClusterMCTruth", &plabels);
  tree.Fill();
  tree.Write();
  mFile->Close();

  mState = 2;
  pc.services().get<ControlService>().readyToQuit(true);
}

DataProcessorSpec getClusterWriterSpec()
{
  return DataProcessorSpec{
    "mft-cluster-writer",
    Inputs{
      InputSpec{ "compClusters", "MFT", "COMPCLUSTERS", 0, Lifetime::Timeframe },
      InputSpec{ "clusters", "MFT", "CLUSTERS", 0, Lifetime::Timeframe },
      InputSpec{ "labels", "MFT", "CLUSTERSMCTR", 0, Lifetime::Timeframe },
      InputSpec{ "ROframes", "MFT", "MFTClusterROF", 0, Lifetime::Timeframe },
      InputSpec{ "MC2ROframes", "MFT", "MFTClusterMC2ROF", 0, Lifetime::Timeframe } },
    Outputs{},
    AlgorithmSpec{ adaptFromTask<ClusterWriter>() },
    Options{
      { "mft-cluster-outfile", VariantType::String, "mftclusters.root", { "Name of the output file" } } }
  };
}

} // namespace MFT
} // namespace o2

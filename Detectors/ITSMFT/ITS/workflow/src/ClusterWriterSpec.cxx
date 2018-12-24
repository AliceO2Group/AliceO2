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
namespace ITS
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

  auto compClusters = pc.inputs().get<const std::vector<o2::ITSMFT::CompClusterExt>>("compClusters");
  auto clusters = pc.inputs().get<const std::vector<o2::ITSMFT::Cluster>>("clusters");
  auto labels = pc.inputs().get<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("labels");
  auto plabels = labels.get();
  auto rofs = pc.inputs().get<const std::vector<o2::ITSMFT::ROFRecord>>("ROframes");
  auto mc2rofs = pc.inputs().get<const std::vector<o2::ITSMFT::MC2ROFRecord>>("MC2ROframes");

  LOG(INFO) << "ITSClusterWriter pulled " << clusters.size() << " clusters, "
            << labels->getIndexedSize() << " MC label objects, in "
            << rofs.size() << " RO frames and "
            << mc2rofs.size() << " MC events";

  mFile->WriteObjectAny(&rofs, "std::vector<o2::ITSMFT::ROFRecord>", "ITSClusterROF");
  mFile->WriteObjectAny(&mc2rofs, "std::vector<o2::ITSMFT::MC2ROFRecord>", "ITSClusterMC2ROF");

  TTree tree("o2sim", "Tree with ITS clusters");
  tree.Branch("ITSClusterComp", &compClusters);
  tree.Branch("ITSCluster", &clusters);
  tree.Branch("ITSClusterMCTruth", &plabels);
  tree.Fill();
  tree.Write();
  mFile->Close();

  mState = 2;
  //pc.services().get<ControlService>().readyToQuit(true);
}

DataProcessorSpec getClusterWriterSpec()
{
  return DataProcessorSpec{
    "its-cluster-writer",
    Inputs{
      InputSpec{ "compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe },
      InputSpec{ "clusters", "ITS", "CLUSTERS", 0, Lifetime::Timeframe },
      InputSpec{ "labels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe },
      InputSpec{ "ROframes", "ITS", "ITSClusterROF", 0, Lifetime::Timeframe },
      InputSpec{ "MC2ROframes", "ITS", "ITSClusterMC2ROF", 0, Lifetime::Timeframe } },
    Outputs{},
    AlgorithmSpec{ adaptFromTask<ClusterWriter>() },
    Options{
      { "its-cluster-outfile", VariantType::String, "o2clus_its.root", { "Name of the output file" } } }
  };
}

} // namespace ITS
} // namespace o2

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

#include "MFTWorkflow/ClusterWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
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

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using CompClusType = std::vector<o2::itsmft::CompClusterExt>;
using PatternsType = std::vector<unsigned char>;
using ClustersType = std::vector<o2::itsmft::Cluster>;
using ROFrameRType = std::vector<o2::itsmft::ROFRecord>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using ROFRecLblT = std::vector<o2::itsmft::MC2ROFRecord>;
using namespace o2::header;

DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  // Spectators for logging
  // this is only to restore the original behavior
  auto clustersSize = std::make_shared<int>(0);
  auto clustersSizeGetter = [clustersSize](ClustersType const& clusters) {
    *clustersSize = clusters.size();
  };
  auto logger = [clustersSize](std::vector<o2::itsmft::ROFRecord> const& rofs) {
    LOG(INFO) << "MFTClusterWriter pulled " << *clustersSize << " clusters, in " << rofs.size() << " RO frames";
  };
  return MakeRootTreeWriterSpec("mft-cluster-writer",
                                "mftclusters.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with MFT clusters"},
                                BranchDefinition<CompClusType>{InputSpec{"compclus", "MFT", "COMPCLUSTERS", 0},
                                                               "MFTClusterComp"},
                                BranchDefinition<PatternsType>{InputSpec{"patterns", "MFT", "PATTERNS", 0},
                                                               "MFTClusterPatt"},
                                // this has been marked to be removed in the original implementation
                                // RSTODO being eliminated
                                BranchDefinition<ClustersType>{InputSpec{"clusters", "MFT", "CLUSTERS", 0},
                                                               "MFTCluster",
                                                               clustersSizeGetter},
                                BranchDefinition<ROFrameRType>{InputSpec{"ROframes", "MFT", "CLUSTERSROF", 0},
                                                               "MFTClustersROF",
                                                               logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "MFT", "CLUSTERSMCTR", 0},
                                                             "MFTClusterMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", "MFT", "CLUSTERSMC2ROF", 0},
                                                             "MFTClustersMC2ROF",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace mft
} // namespace o2

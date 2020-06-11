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

#include "ITSWorkflow/ClusterWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace its
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
  auto compClustersSize = std::make_shared<int>(0);
  auto compClustersSizeGetter = [compClustersSize](CompClusType const& compClusters) {
    *compClustersSize = compClusters.size();
  };
  auto logger = [compClustersSize](std::vector<o2::itsmft::ROFRecord> const& rofs) {
    LOG(INFO) << "ITSClusterWriter pulled " << *compClustersSize << " clusters, in " << rofs.size() << " RO frames";
  };
  return MakeRootTreeWriterSpec("its-cluster-writer",
                                "o2clus_its.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with ITS clusters"},
                                BranchDefinition<CompClusType>{InputSpec{"compclus", "ITS", "COMPCLUSTERS", 0},
                                                               "ITSClusterComp",
                                                               compClustersSizeGetter},
                                BranchDefinition<PatternsType>{InputSpec{"patterns", "ITS", "PATTERNS", 0},
                                                               "ITSClusterPatt"},
                                // this has been marked to be removed in the original implementation
                                // RSTODO being eliminated
                                BranchDefinition<ClustersType>{InputSpec{"clusters", "ITS", "CLUSTERS", 0},
                                                               "ITSCluster"},
                                BranchDefinition<ROFrameRType>{InputSpec{"ROframes", "ITS", "ClusterROF", 0},
                                                               "ITSClustersROF",
                                                               logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "ITS", "CLUSTERSMCTR", 0},
                                                             "ITSClusterMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", "ITS", "ClusterMC2ROF", 0},
                                                             "ITSClustersMC2ROF",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace its
} // namespace o2

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

#include "ITS3Workflow/ClusterWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsITS3/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace its3
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using CompClusType = std::vector<o2::its3::CompClusterExt>;
using PatternsType = std::vector<unsigned char>;
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
    LOG(INFO) << "ITS3ClusterWriter pulled " << *compClustersSize << " clusters, in " << rofs.size() << " RO frames";
  };
  return MakeRootTreeWriterSpec("its3-cluster-writer",
                                "o2clus_it3.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with ITS clusters"},
                                BranchDefinition<CompClusType>{InputSpec{"compclus", "IT3", "COMPCLUSTERS", 0},
                                                               "IT3ClusterComp",
                                                               compClustersSizeGetter},
                                BranchDefinition<PatternsType>{InputSpec{"patterns", "IT3", "PATTERNS", 0},
                                                               "IT3ClusterPatt"},
                                BranchDefinition<ROFrameRType>{InputSpec{"ROframes", "IT3", "CLUSTERSROF", 0},
                                                               "IT3ClustersROF",
                                                               logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "IT3", "CLUSTERSMCTR", 0},
                                                             "IT3ClusterMCTruth",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""},
                                BranchDefinition<ROFRecLblT>{InputSpec{"MC2ROframes", "IT3", "CLUSTERSMC2ROF", 0},
                                                             "IT3ClustersMC2ROF",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             ""})();
}

} // namespace its
} // namespace o2

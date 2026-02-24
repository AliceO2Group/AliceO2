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

/// @file   ClusterWriterSpec.cxx

#include <vector>

#include "TRKWorkflow/ClusterWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2::trk
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using ClustersType = std::vector<o2::trk::Cluster>;
using PatternsType = std::vector<unsigned char>;
using ROFrameType = std::vector<o2::trk::ROFRecord>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using ROFRecLblType = std::vector<o2::trk::MC2ROFRecord>;

DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  auto clustersSize = std::make_shared<int>(0);
  auto clustersSizeGetter = [clustersSize](ClustersType const& clusters) {
    *clustersSize = clusters.size();
  };
  auto logger = [clustersSize](ROFrameType const& rofs) {
    LOG(info) << "TRKClusterWriter pulled " << *clustersSize << " clusters, in " << rofs.size() << " RO frames";
  };

  return MakeRootTreeWriterSpec("trk-cluster-writer",
                                "o2clus_trk.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with TRK clusters"},
                                BranchDefinition<ClustersType>{InputSpec{"compclus", "TRK", "COMPCLUSTERS", 0},
                                                               "TRKClusterComp",
                                                               clustersSizeGetter},
                                BranchDefinition<PatternsType>{InputSpec{"patterns", "TRK", "PATTERNS", 0},
                                                               "TRKClusterPatt"},
                                BranchDefinition<ROFrameType>{InputSpec{"ROframes", "TRK", "CLUSTERSROF", 0},
                                                              "TRKClustersROF",
                                                              logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", "TRK", "CLUSTERSMCTR", 0},
                                                             "TRKClusterMCTruth",
                                                             (useMC ? 1 : 0)},
                                BranchDefinition<ROFRecLblType>{InputSpec{"MC2ROframes", "TRK", "CLUSTERSMC2ROF", 0},
                                                                "TRKClustersMC2ROF",
                                                                (useMC ? 1 : 0)})();
}

} // namespace o2::trk

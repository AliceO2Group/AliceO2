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

/// @file   TODClusterWriterSpec.cxx

#include "TOFWorkflowIO/TOFClusterWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsTOF/Cluster.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using OutputType = std::vector<o2::tof::Cluster>;
using MultType = std::vector<int>;
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getTOFClusterWriterSpec(bool useMC)
{
  // Spectators for logging
  auto logger = [](OutputType const& indata) {
    LOG(debug) << "RECEIVED CLUSTERS SIZE " << indata.size();
  };
  auto loggerMult = [](MultType const& inmult) {
    LOG(debug) << "RECEIVED N BC SIZE " << inmult.size();
  };
  auto loggerMCLabels = [](LabelsType const& labeldata) {
    LOG(debug) << "TOF GOT " << labeldata.getNElements() << " LABELS ";
  };
  return MakeRootTreeWriterSpec("TOFClusterWriter",
                                "tofclusters.root",
                                "o2sim",
                                BranchDefinition<OutputType>{InputSpec{"clusters", gDataOriginTOF, "CLUSTERS", 0},
                                                             "TOFCluster",
                                                             "tofclusters-branch-name",
                                                             1,
                                                             logger},
                                BranchDefinition<MultType>{InputSpec{"clustersMult", gDataOriginTOF, "CLUSTERSMULT", 0},
                                                           "TOFClusterMult",
                                                           "tofclustersmult-branch-name",
                                                           1,
                                                           loggerMult},
                                BranchDefinition<LabelsType>{InputSpec{"labels", gDataOriginTOF, "CLUSTERSMCTR", 0},
                                                             "TOFClusterMCTruth",
                                                             "clusterlabels-branch-name",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             loggerMCLabels})();
}
} // namespace tof
} // namespace o2

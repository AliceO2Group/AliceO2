// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
using LabelsType = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getTOFClusterWriterSpec(bool useMC)
{
  // Spectators for logging
  auto logger = [](OutputType const& indata) {
    LOG(DEBUG) << "RECEIVED CLUSTERS SIZE " << indata.size();
  };
  auto loggerMCLabels = [](LabelsType const& labeldata) {
    LOG(DEBUG) << "TOF GOT " << labeldata.getNElements() << " LABELS ";
  };
  return MakeRootTreeWriterSpec("TOFClusterWriter",
                                "tofclusters.root",
                                "o2sim",
                                BranchDefinition<OutputType>{InputSpec{"clusters", gDataOriginTOF, "CLUSTERS", 0},
                                                             "TOFCluster",
                                                             "tofclusters-branch-name",
                                                             1,
                                                             logger},
                                BranchDefinition<LabelsType>{InputSpec{"labels", gDataOriginTOF, "CLUSTERSMCTR", 0},
                                                             "TOFClusterMCTruth",
                                                             "clusterlabels-branch-name",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             loggerMCLabels})();
}
} // namespace tof
} // namespace o2

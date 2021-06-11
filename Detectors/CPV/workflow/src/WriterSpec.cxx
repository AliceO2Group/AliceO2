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

#include "CPVWorkflow/WriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsCPV/Cluster.h"
#include "DataFormatsCPV/TriggerRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace cpv
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using ClusType = std::vector<o2::cpv::Cluster>;
using DigitType = std::vector<o2::cpv::Digit>;
using TriggerRecordType = std::vector<o2::cpv::TriggerRecord>;
using MCLabelType = o2::dataformats::MCTruthContainer<MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getClusterWriterSpec(bool useMC)
{
  // Spectators for logging
  // this is only to restore the original behavior
  auto ClustersSize = std::make_shared<int>(0);
  auto ClustersSizeGetter = [ClustersSize](ClusType const& Clusters) {
    *ClustersSize = Clusters.size();
  };

  if (useMC) {
    return MakeRootTreeWriterSpec("cpv-cluster-writer",
                                  "cpvclusters.root",
                                  MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CPV clusters"},
                                  BranchDefinition<ClusType>{InputSpec{"clus", "CPV", "CLUSTERS", 0},
                                                             "CPVCluster", ClustersSizeGetter},
                                  BranchDefinition<TriggerRecordType>{InputSpec{"clusRecs", "CPV", "CLUSTERTRIGRECS", 0},
                                                                      "CPVClusterTrigRec"},
                                  BranchDefinition<MCLabelType>{InputSpec{"clusMC", "CPV", "CLUSTERTRUEMC", 0},
                                                                "CPVClusterTrueMC"})();
  } else {
    return MakeRootTreeWriterSpec("cpv-cluster-writer",
                                  "cpvclusters.root",
                                  MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CPV clusters"},
                                  BranchDefinition<ClusType>{InputSpec{"clus", "CPV", "CLUSTERS", 0},
                                                             "CPVCluster", ClustersSizeGetter},
                                  BranchDefinition<TriggerRecordType>{InputSpec{"clusRecs", "CPV", "CLUSTERTRIGRECS", 0},
                                                                      "CPVClusterTrigRec"})();
  }
}

DataProcessorSpec getDigitWriterSpec(bool useMC)
{
  // Spectators for logging
  // this is only to restore the original behavior
  auto DigitsSize = std::make_shared<int>(0);
  auto DigitsSizeGetter = [DigitsSize](DigitType const& Digits) {
    *DigitsSize = Digits.size();
  };

  if (useMC) {
    return MakeRootTreeWriterSpec("cpv-digit-writer",
                                  "cpvdigits.root",
                                  MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CPV digits"},
                                  BranchDefinition<DigitType>{InputSpec{"CPVDigit", "CPV", "DIGITS", 0},
                                                              "CPVDigit", DigitsSizeGetter},
                                  BranchDefinition<TriggerRecordType>{InputSpec{"CPVDigitTrigRecords", "CPV", "DIGITTRIGREC", 0},
                                                                      "CPVDigitTrigRecords"},
                                  BranchDefinition<MCLabelType>{InputSpec{"clusMC", "CPV", "DIGITSMCTR", 0},
                                                                "CPVDigitMCTruth"})();
  } else {
    return MakeRootTreeWriterSpec("cpv-digit-writer",
                                  "cpvdigits.root",
                                  MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with CPV digits"},
                                  BranchDefinition<DigitType>{InputSpec{"CPVDigit", "CPV", "DIGITS", 0},
                                                              "CPVDigit", DigitsSizeGetter},
                                  BranchDefinition<TriggerRecordType>{InputSpec{"CPVDigitTrigRecords", "CPV", "DIGITTRIGREC", 0},
                                                                      "CPVDigitTrigRecords"})();
  }
}

} // namespace cpv
} // namespace o2
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

#include "HMPIDWorkflow/ClustersWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Framework/InputSpec.h"
#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "DataFormatsHMP/Cluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace hmpid
{

template <typename T>
using BranchDefinition = framework::MakeRootTreeWriterSpec::BranchDefinition<T>;

o2::framework::DataProcessorSpec getClusterWriterSpec()
{
  using InputSpec = framework::InputSpec;
  using MakeRootTreeWriterSpec = framework::MakeRootTreeWriterSpec;

  return MakeRootTreeWriterSpec("HMPClustersWriter",
                                "hmpidclusters.root",
                                "o2hmp",
                                1,
                                BranchDefinition<std::vector<o2::hmpid::Cluster>>{InputSpec{"hmpclusterinput", "HMP", "CLUSTERS"}, "HMPIDclusters"},
                                BranchDefinition<std::vector<o2::hmpid::Trigger>>{InputSpec{"hmpinteractionrecords", "HMP", "INTRECORDS1"}, "InteractionRecords"})();
}

} // end namespace hmpid
} // end namespace o2

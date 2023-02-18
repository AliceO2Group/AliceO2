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

#include "MCHIO/ClusterWriterSpec.h"

#include <vector>
#include "Framework/Logger.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"

using namespace o2::framework;

namespace o2::mch
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getClusterWriterSpec(bool useMC, const char* specName, bool global, bool digits)
{
  auto clusterDescription = global ? header::DataDescription{"GLOBALCLUSTERS"} : header::DataDescription{"CLUSTERS"};
  return MakeRootTreeWriterSpec(specName,
                                "mchclusters.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree MCH Clusters"},
                                BranchDefinition<std::vector<Cluster>>{InputSpec{"clusters", "MCH", clusterDescription}, "clusters"},
                                BranchDefinition<std::vector<ROFRecord>>{InputSpec{"clusterrofs", "MCH", "CLUSTERROFS"}, "clusterrofs"},
                                BranchDefinition<std::vector<Digit>>{InputSpec{"clusterdigits", "MCH", "CLUSTERDIGITS"}, "clusterdigits", digits ? 1 : 0},
                                BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{InputSpec{"clusterlabels", "MCH", "CLUSTERLABELS"}, "clusterlabels", useMC ? 1 : 0})();
}

} // namespace o2::mch

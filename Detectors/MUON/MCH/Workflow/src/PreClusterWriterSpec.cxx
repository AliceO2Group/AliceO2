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

#include "MCHWorkflow/PreClusterWriterSpec.h"

#include <vector>
#include "Framework/Logger.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "DataFormatsMCH/Digit.h"
#include "MCHBase/PreCluster.h"

using namespace o2::framework;

namespace o2::mch
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getPreClusterWriterSpec(bool useMC, const char* specName)
{
  return MakeRootTreeWriterSpec(specName,
                                "mchpreclusters.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree MCH PreClusters"},
                                BranchDefinition<std::vector<ROFRecord>>{InputSpec{"preclusterrofs", "MCH", "PRECLUSTERROFS"}, "preclusterrofs"},
                                BranchDefinition<std::vector<PreCluster>>{InputSpec{"preclusters", "MCH", "PRECLUSTERS"}, "preclusters"},
                                BranchDefinition<std::vector<Digit>>{InputSpec{"preclusterdigits", "MCH", "PRECLUSTERDIGITS"}, "preclusterdigits"},
                                BranchDefinition<dataformats::MCTruthContainer<MCCompLabel>>{InputSpec{"preclusterlabels", "MCH", "PRECLUSTERLABELS"}, "preclusterlabels", useMC ? 1 : 0})();
}

} // namespace o2::mch

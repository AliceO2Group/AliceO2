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

/// @file   TOFMatchedWriterSpec.cxx

#include "HMPIDWorkflow/HMPMatchedWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "ReconstructionDataFormats/MatchInfoHMP.h"
#include "ReconstructionDataFormats/MatchingType.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"

#include "DataFormatsHMP/Cluster.h"
#include "CommonUtils/StringUtils.h"
#include <sstream>

using namespace o2::framework;

namespace o2
{
namespace hmpid
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using MatchInfo = std::vector<o2::dataformats::MatchInfoHMP>;
using LabelsType = std::vector<o2::MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getHMPMatchedWriterSpec(bool useMC, const char* outdef) //, bool writeTracks, int mode, bool strict)
{

  const char* taskName = "HMPMatchedWriter";

  return MakeRootTreeWriterSpec(taskName,
                                outdef,
                                "matchHMP",
                                BranchDefinition<MatchInfo>{InputSpec{"hmpmatching", gDataOriginHMP, "MATCHES", 0},
                                                            "HMPMatchInfo",
                                                            "HMPMatchInfo-branch-name"},
                                BranchDefinition<LabelsType>{InputSpec{"matchhmplabels", gDataOriginHMP, "MCLABELS", 0},
                                                             "MatchHMPMCTruth",
                                                             "MatchHMPMCTruth-branch-name",
                                                             (useMC ? 1 : 0)})();
}
} // namespace hmpid
} // namespace o2

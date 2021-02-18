// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFMatchedWriterSpec.cxx

#include "TOFWorkflow/TOFMatchedWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"
#include "CommonUtils/StringUtils.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using OutputType = std::vector<o2::dataformats::MatchInfoTOF>;
using LabelsType = std::vector<o2::MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getTOFMatchedWriterSpec(bool useMC, const char* outdef)
{
  // spectators for logging
  auto loggerMatched = [](OutputType const& indata) {
    LOG(INFO) << "RECEIVED MATCHED SIZE " << indata.size();
  };
  auto loggerTofLabels = [](LabelsType const& labeltof) {
    LOG(INFO) << "TOF LABELS GOT " << labeltof.size() << " LABELS ";
  };
  auto loggerTpcLabels = [](LabelsType const& labeltpc) {
    LOG(INFO) << "TPC LABELS GOT " << labeltpc.size() << " LABELS ";
  };
  auto loggerItsLabels = [](LabelsType const& labelits) {
    LOG(INFO) << "ITS LABELS GOT " << labelits.size() << " LABELS ";
  };
  // TODO: there was a comment in the original implementation:
  // RS why do we need to repead ITS/TPC labels ?
  // They can be extracted from TPC-ITS matches
  return MakeRootTreeWriterSpec("TOFMatchedWriter",
                                outdef,
                                "matchTOF",
                                BranchDefinition<OutputType>{InputSpec{"tofmatching", gDataOriginTOF, "MATCHINFOS", 0},
                                                             "TOFMatchInfo",
                                                             "TOFMatchInfo-branch-name",
                                                             1,
                                                             loggerMatched},
                                BranchDefinition<LabelsType>{InputSpec{"matchtoflabels", gDataOriginTOF, "MATCHTOFINFOSMC", 0},
                                                             "MatchTOFMCTruth",
                                                             "MatchTOFMCTruth-branch-name",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             loggerTofLabels},
                                BranchDefinition<LabelsType>{InputSpec{"matchtpclabels", gDataOriginTOF, "MATCHTPCINFOSMC", 0},
                                                             "MatchTPCMCTruth",
                                                             "MatchTPCMCTruth-branch-name",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             loggerTpcLabels},
                                BranchDefinition<LabelsType>{InputSpec{"matchitslabels", gDataOriginTOF, "MATCHITSINFOSMC", 0},
                                                             "MatchITSMCTruth",
                                                             "MatchITSMCTruth-branch-name",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             loggerItsLabels})();
}
} // namespace tof
} // namespace o2

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

#include "TOFWorkflowIO/TOFMatchedWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
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
using MatchInfo = std::vector<o2::dataformats::MatchInfoTOF>;
using TrackInfo = std::vector<o2::dataformats::TrackTPCTOF>;
using LabelsType = std::vector<o2::MCCompLabel>;
using namespace o2::header;

DataProcessorSpec getTOFMatchedWriterSpec(bool useMC, const char* outdef, bool writeTOFTPC)
{
  // spectators for logging
  auto loggerMatched = [](MatchInfo const& indata) {
    LOG(INFO) << "RECEIVED MATCHED SIZE " << indata.size();
  };
  auto loggerTofLabels = [](LabelsType const& labeltof) {
    LOG(INFO) << "TOF LABELS GOT " << labeltof.size() << " LABELS ";
  };
  o2::header::DataDescription ddMatchInfo{"MATCHINFOS"}, ddMatchInfo_tpc{"MATCHINFOS_TPC"},
    ddMCMatchTOF{"MCMATCHTOF"}, ddMCMatchTOF_tpc{"MCMATCHTOF_TPC"};

  return MakeRootTreeWriterSpec(writeTOFTPC ? "TOFMatchedWriter_TPC" : "TOFMatchedWriter",
                                outdef,
                                "matchTOF",
                                BranchDefinition<MatchInfo>{InputSpec{"tofmatching", gDataOriginTOF, writeTOFTPC ? ddMatchInfo_tpc : ddMatchInfo, 0},
                                                            "TOFMatchInfo",
                                                            "TOFMatchInfo-branch-name",
                                                            1,
                                                            loggerMatched},
                                BranchDefinition<TrackInfo>{InputSpec{"tpctofTracks", gDataOriginTOF, "TOFTRACKS_TPC", 0},
                                                            "TPCTOFTracks",
                                                            "TPCTOFTracks-branch-name",
                                                            writeTOFTPC},
                                BranchDefinition<LabelsType>{InputSpec{"matchtoflabels", gDataOriginTOF, writeTOFTPC ? ddMCMatchTOF_tpc : ddMCMatchTOF, 0},
                                                             "MatchTOFMCTruth",
                                                             "MatchTOFMCTruth-branch-name",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             loggerTofLabels})();
}
} // namespace tof
} // namespace o2

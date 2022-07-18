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

#include "TOFWorkflowIO/TOFMatchedWriterSpec.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/MatchingType.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"
#include "CommonUtils/StringUtils.h"
#include <sstream>

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

DataProcessorSpec getTOFMatchedWriterSpec(bool useMC, const char* outdef, bool writeTracks, int mode, bool strict)
{
  // spectators for logging
  auto loggerMatched = [](MatchInfo const& indata) {
    LOG(debug) << "RECEIVED MATCHED SIZE " << indata.size();
  };
  auto loggerTofLabels = [](LabelsType const& labeltof) {
    LOG(debug) << "TOF LABELS GOT " << labeltof.size() << " LABELS ";
  };
  //  o2::header::DataDescription ddMatchInfo{"MTC_ITSTPC"}, ddMatchInfo_tpc{"MTC_TPC"},
  //    ddMCMatchTOF{"MCMTC_ITSTPC"}, ddMCMatchTOF_tpc{"MCMTC_TPC"};

  o2::header::DataDescription ddMatchInfo[4] = {{"MTC_TPC"}, {"MTC_ITSTPC"}, {"MTC_TPCTRD"}, {"MTC_ITSTPCTRD"}};
  o2::header::DataDescription ddMCMatchTOF[4] = {{"MCMTC_TPC"}, {"MCMTC_ITSTPC"}, {"MCMTC_TPCTRD"}, {"MCMTC_ITSTPCTRD"}};

  uint32_t ss = o2::globaltracking::getSubSpec(strict ? o2::globaltracking::MatchingType::Strict : o2::globaltracking::MatchingType::Standard);

  const char* match_name[4] = {"TOFMatchedWriter_TPC", "TOFMatchedWriter_ITSTPC", "TOFMatchedWriter_TPCTRD", "TOFMatchedWriter_ITSTPCTRD"};
  const char* match_name_strict[4] = {"TOFMatchedWriter_TPC_str", "TOFMatchedWriter_ITSTPC_str", "TOFMatchedWriter_TPCTRD_str", "TOFMatchedWriter_ITSTPCTRD_str"};

  const char* taskName = match_name[mode];
  if (strict) {
    taskName = match_name_strict[mode];
  }

  // inputBindings better be unique for each data spec, otherwise
  // they can not be "combined" into a single DPL device
  std::stringstream inputBinding1, inputBinding2, inputBinding3;
  inputBinding1 << "tofmatching_" << mode;
  inputBinding2 << "tpctofTracks_" << mode;
  inputBinding3 << "matchtoflabels_" << mode;

  return MakeRootTreeWriterSpec(taskName,
                                outdef,
                                "matchTOF",
                                BranchDefinition<MatchInfo>{InputSpec{inputBinding1.str().c_str(), gDataOriginTOF, ddMatchInfo[mode], ss},
                                                            "TOFMatchInfo",
                                                            "TOFMatchInfo-branch-name",
                                                            1,
                                                            loggerMatched},
                                BranchDefinition<TrackInfo>{InputSpec{inputBinding2.str().c_str(), gDataOriginTOF, "TOFTRACKS_TPC", ss},
                                                            "TPCTOFTracks",
                                                            "TPCTOFTracks-branch-name",
                                                            writeTracks},
                                BranchDefinition<LabelsType>{InputSpec{inputBinding3.str().c_str(), gDataOriginTOF, ddMCMatchTOF[mode], ss},
                                                             "MatchTOFMCTruth",
                                                             "MatchTOFMCTruth-branch-name",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             loggerTofLabels})();
}
} // namespace tof
} // namespace o2

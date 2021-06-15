// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFCalibWriterSpec.cxx

#include "TOFWorkflowIO/TOFCalibWriterSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "Headers/DataHeader.h"
#include <SimulationDataFormat/MCCompLabel.h>
#include <SimulationDataFormat/MCTruthContainer.h>
#include "TTree.h"
#include "TBranch.h"
#include "TFile.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "DataFormatsTOF/Cluster.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;
using CalibInfosType = std::vector<o2::dataformats::CalibInfoTOF>;
DataProcessorSpec getTOFCalibWriterSpec(const char* outdef, bool toftpc)
{
  // A spectator for logging
  auto logger = [](CalibInfosType const& indata) {
    LOG(INFO) << "RECEIVED MATCHED SIZE " << indata.size();
  };
  o2::header::DataDescription ddCalib{"CALIBDATA"}, ddCalib_tpc{"CALIBDATA_TPC"};
  return MakeRootTreeWriterSpec("TOFCalibWriter",
                                outdef,
                                "calibTOF",
                                BranchDefinition<CalibInfosType>{InputSpec{"input", o2::header::gDataOriginTOF, toftpc ? ddCalib_tpc : ddCalib, 0},
                                                                 "TOFCalibInfo",
                                                                 "calibinfo-branch-name",
                                                                 1,
                                                                 logger})();
}

} // namespace tof
} // namespace o2

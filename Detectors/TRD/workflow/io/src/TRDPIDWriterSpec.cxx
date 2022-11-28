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

/// @file  TRDPIDWriterSpec.cxx
/// @author Felix Schlepper

#include <vector>
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"
#include "DataFormatsTRD/PID.h"
#include "TRDWorkflowIO/TRDPIDWriterSpec.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

using LabelsType = std::vector<o2::MCCompLabel>;

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTRDPIDGlobalWriterSpec(bool useMC)
{
  // A spectator to store the size of the data array for the logger below
  auto pidSize = std::make_shared<size_t>();
  auto pidLogger = [pidSize](std::vector<o2::trd::PIDValue> const& def) {
    *pidSize = def.size();
  };

  return MakeRootTreeWriterSpec("trd-pid-writer-itstpc",
                                "trdpid_itstpc.root",
                                "pidTRD",
                                BranchDefinition<std::vector<o2::trd::PIDValue>>{InputSpec{"pid", o2::header::gDataOriginTRD, "TRDPID_ITSTPC", 0},
                                                                                 "pid",
                                                                                 "pid-branch-name",
                                                                                 1,
                                                                                 pidLogger},
                                BranchDefinition<std::vector<o2::trd::TrackTriggerRecord>>{InputSpec{"trackTrig", o2::header::gDataOriginTRD, "TRGREC_ITSTPC", 0},
                                                                                           "trgrec",
                                                                                           "trgrec-branch-name",
                                                                                           1},
                                BranchDefinition<LabelsType>{InputSpec{"trdlabels", o2::header::gDataOriginTRD, "MCLB_ITSTPC_TRD", 0},
                                                             "labelsTRD",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "trdlabels-branch-name"},
                                BranchDefinition<LabelsType>{InputSpec{"matchitstpclabels", o2::header::gDataOriginTRD, "MCLB_ITSTPC", 0},
                                                             "labels",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "labels-branch-name"})();
}

DataProcessorSpec
  getTRDPIDTPCWriterSpec(bool useMC)
{
  // A spectator to store the size of the data array for the logger below
  auto pidSize = std::make_shared<size_t>();
  auto pidLogger = [pidSize](std::vector<o2::trd::PIDValue> const& def) {
    *pidSize = def.size();
  };

  return MakeRootTreeWriterSpec("trd-pid-writer-tpc",
                                "trdpid_tpc.root",
                                "pidTRD",
                                BranchDefinition<std::vector<o2::trd::PIDValue>>{InputSpec{"pid", o2::header::gDataOriginTRD, "TRDPID_TPC", 0},
                                                                                 "pid",
                                                                                 "pid-branch-name",
                                                                                 1,
                                                                                 pidLogger},
                                BranchDefinition<std::vector<o2::trd::TrackTriggerRecord>>{InputSpec{"trackTrig", o2::header::gDataOriginTRD, "TRGREC_TPC", 0},
                                                                                           "trgrec",
                                                                                           "trgrec-branch-name",
                                                                                           1},
                                BranchDefinition<LabelsType>{InputSpec{"trdlabels", o2::header::gDataOriginTRD, "MCLB_TPC_TRD", 0},
                                                             "labelsTRD",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "trdlabels-branch-name"},
                                BranchDefinition<LabelsType>{InputSpec{"matchtpclabels", o2::header::gDataOriginTRD, "MCLB_TPC", 0},
                                                             "labels",
                                                             (useMC ? 1 : 0), // one branch if mc labels enabled
                                                             "labels-branch-name"})();
}

} // namespace trd
} // namespace o2

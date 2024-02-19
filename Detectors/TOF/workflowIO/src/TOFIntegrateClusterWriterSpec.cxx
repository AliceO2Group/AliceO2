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

/// @file  TOFIntegrateClusterWriterSpec.cxx

#include <vector>
#include "TOFWorkflowIO/TOFIntegrateClusterWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TFIDInfo.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTOFIntegrateClusterWriterSpec()
{
  return MakeRootTreeWriterSpec("tof-currents-writer",
                                "o2currents_tof.root",
                                "itofc",
                                BranchDefinition<std::vector<float>>{InputSpec{"itofcn", o2::header::gDataOriginTOF, "ITOFCN", 0, Lifetime::Timeframe}, "ITOFCN", 1},
                                BranchDefinition<std::vector<float>>{InputSpec{"itofcq", o2::header::gDataOriginTOF, "ITOFCQ", 0, Lifetime::Timeframe}, "ITOFCQ", 1},
                                BranchDefinition<o2::dataformats::TFIDInfo>{InputSpec{"itoftfid", o2::header::gDataOriginTOF, "ITOFTFID", 0, Lifetime::Timeframe}, "tfID", 1})();
}

} // namespace tof
} // namespace o2

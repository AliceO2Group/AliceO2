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

/// @file  TPCIntegrateClusterWriterSpec.cxx

#include <vector>
#include "TPCWorkflow/TPCTriggerWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "DataFormatsTPC/ZeroSuppression.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTPCTriggerWriterSpec()
{
  return MakeRootTreeWriterSpec("tpc-trigger-writer",
                                "tpctriggers.root",
                                "triggers",
                                BranchDefinition<std::vector<o2::tpc::TriggerInfoDLBZS>>{InputSpec{"trig", o2::header::gDataOriginTPC, "TRIGGERWORDS", 0}, "Triggers"})();
}

} // namespace tpc
} // namespace o2

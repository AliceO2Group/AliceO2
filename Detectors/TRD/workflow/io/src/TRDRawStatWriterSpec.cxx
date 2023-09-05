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

/// @file  TRDRawStatWriterSpec.cxx

#include <vector>
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "TRDWorkflowIO/TRDRawStatWriterSpec.h"
#include "DataFormatsTRD/TriggerRecord.h"
#include "DataFormatsTRD/RawDataStats.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTRDRawStatWriterSpec(bool linkStats)
{

  return MakeRootTreeWriterSpec("trd-rawstat-writer",
                                "trdrawstats.root",
                                "stats",
                                BranchDefinition<std::vector<TriggerRecord>>{InputSpec{"trigrec", "TRD", "TRKTRGRD"}, "trigRec", 1},
                                BranchDefinition<std::vector<DataCountersPerTrigger>>{InputSpec{"linkstats", "TRD", "LINKSTATS"}, "linkStats", (linkStats ? 1 : 0)},
                                BranchDefinition<TRDDataCountersPerTimeFrame>{InputSpec{"tfstats", "TRD", "RAWSTATS"}, "tfStats", 1})();
}

} // namespace trd
} // namespace o2

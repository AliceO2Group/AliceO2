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

/// @file  TPCTimeSeriesWriterSpec.cxx

// #include <vector>
#include "TPCWorkflow/TPCTimeSeriesWriterSpec.h"
#include "TPCWorkflow/TPCTimeSeriesSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/TFIDInfo.h"

#include "DetectorsCalibration/IntegratedClusterCalibrator.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getTPCTimeSeriesWriterSpec()
{
  return MakeRootTreeWriterSpec("tpc-time-series-writer",
                                "o2_timeseries_tpc.root",
                                "treeTimeSeries",
                                BranchDefinition<TimeSeriesITSTPC>{InputSpec{"timeseries", o2::header::gDataOriginTPC, getDataDescriptionTimeSeries(), 0, Lifetime::Sporadic}, "TimeSeries", 1},
                                BranchDefinition<o2::dataformats::TFIDInfo>{InputSpec{"itpctfid", o2::header::gDataOriginTPC, getDataDescriptionTPCTimeSeriesTFId(), 0, Lifetime::Sporadic}, "tfID", 1})();
}

} // namespace tpc
} // namespace o2

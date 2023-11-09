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

#ifndef O2_TPC_TPCTIMESERIESSPEC_SPEC
#define O2_TPC_TPCTIMESERIESSPEC_SPEC

#include "Framework/DataProcessorSpec.h"
#include "DetectorsBase/Propagator.h"

namespace o2
{
namespace tpc
{
static constexpr header::DataDescription getDataDescriptionTimeSeries() { return header::DataDescription{"TIMESERIES"}; }
static constexpr header::DataDescription getDataDescriptionTPCTimeSeriesTFId() { return header::DataDescription{"ITPCTSTFID"}; }

o2::framework::DataProcessorSpec getTPCTimeSeriesSpec(const bool disableWriter, const o2::base::Propagator::MatCorrType matType, const bool enableUnbinnedWriter);

} // end namespace tpc
} // end namespace o2

#endif

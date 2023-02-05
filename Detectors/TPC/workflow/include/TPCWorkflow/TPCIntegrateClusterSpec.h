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

#ifndef O2_TPC_TPCINTEGRATECLUSTERSPEC_SPEC
#define O2_TPC_TPCINTEGRATECLUSTERSPEC_SPEC

#include "Framework/DataProcessorSpec.h"

namespace o2
{
namespace tpc
{
static constexpr header::DataDescription getDataDescriptionTPCC() { return header::DataDescription{"ITPCC"}; }
static constexpr header::DataDescription getDataDescriptionTPCTFId() { return header::DataDescription{"ITPCTFID"}; }

o2::framework::DataProcessorSpec getTPCIntegrateClusterSpec(const bool disableWriter);

} // end namespace tpc
} // end namespace o2

#endif

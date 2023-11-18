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

/// @file   TPCITSMatchingSpec.h

#ifndef O2_MATCHING_TPCITS_SPEC
#define O2_MATCHING_TPCITS_SPEC

#include "ReconstructionDataFormats/GlobalTrackID.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
struct CorrectionMapsLoaderGloOpts;
}
namespace globaltracking
{
/// create a processor spec
framework::DataProcessorSpec getTPCITSMatchingSpec(o2::dataformats::GlobalTrackID::mask_t src, bool useFT0, bool calib, bool skipTPCOnly, bool useMC, const o2::tpc::CorrectionMapsLoaderGloOpts& sclOpts);

} // namespace globaltracking
} // namespace o2

#endif /* O2_TRACKWRITER_TPCITS */

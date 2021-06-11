// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CosmicsMatchingSpec.h

#ifndef O2_COSMICS_MATCHING_SPEC
#define O2_COSMICS_MATCHING_SPEC

#include "Framework/DataProcessorSpec.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

/// create a processor spec
framework::DataProcessorSpec getCosmicsMatchingSpec(o2::dataformats::GlobalTrackID::mask_t src, bool useMC);

} // namespace globaltracking
} // namespace o2

#endif /* O2_TRACKWRITER_TPCITS */

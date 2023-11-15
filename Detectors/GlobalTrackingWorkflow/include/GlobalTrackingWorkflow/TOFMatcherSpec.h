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

/// @file   TOFMatcherSpec.h

#ifndef O2_TOF_MATCHER_SPEC
#define O2_TOF_MATCHER_SPEC

#include "Framework/DataProcessorSpec.h"
#include "ReconstructionDataFormats/MatchInfoTOFReco.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

/// create a processor spec
framework::DataProcessorSpec getTOFMatcherSpec(o2::dataformats::GlobalTrackID::mask_t src, bool useMC, bool useFIT, bool tpcRefit, bool strict, float extratolerancetrd, bool pushMatchable, int lumiType);

} // namespace globaltracking
} // namespace o2

#endif /* O2_TOF_MATCHER_SPEC */

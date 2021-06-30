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

/// \file LHCConstants.h
/// \brief Header to collect LHC related constants
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_LHCCONSTANTS_H_
#define ALICEO2_LHCCONSTANTS_H_

namespace o2
{
namespace constants
{
namespace lhc
{
// LHC Beam1 and Beam2 definitions
enum BeamDirection : int { BeamClockWise,
                           BeamAntiClockWise,
                           NBeamDirections };
static constexpr int LHCMaxBunches = 3564;                              // max N bunches
static constexpr double LHCRFFreq = 400.789e6;                          // LHC RF frequency in Hz
static constexpr double LHCBunchSpacingNS = 10 * 1.e9 / LHCRFFreq;      // bunch spacing in ns (10 RFbuckets)
static constexpr double LHCOrbitNS = LHCMaxBunches * LHCBunchSpacingNS; // orbit duration in ns
static constexpr double LHCRevFreq = 1.e9 / LHCOrbitNS;                 // revolution frequency
static constexpr double LHCBunchSpacingMUS = LHCBunchSpacingNS * 1e-3;  // bunch spacing in \mus (10 RFbuckets)
static constexpr double LHCOrbitMUS = LHCOrbitNS * 1e-3;                // orbit duration in \mus

static constexpr unsigned int MaxNOrbits = 0xffffffff;
} // namespace lhc
} // namespace constants
} // namespace o2

#endif

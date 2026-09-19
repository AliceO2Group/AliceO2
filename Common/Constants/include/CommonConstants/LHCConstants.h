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

#include "GPUCommonDouble.h"

#include "GPUCommonDef.h"

namespace o2
{
namespace constants
{
namespace lhc
{
// LHC Beam1 and Beam2 definitions
enum BeamDirection : int { BeamA, // beamA = beam 0,
                           BeamC, // beamC = beam 1
                           NBeamDirections,
                           InteractingBC = -1 // as used in the BunchFilling class
};
GPUglobalconstexpr() int LHCMaxBunches = 3564;                              // max N bunches
GPUglobalconstexpr() o2::gpu::GPUdoubleValue LHCRFFreq = 400.789e6;                          // LHC RF frequency in Hz
GPUglobalconstexpr() o2::gpu::GPUdoubleValue LHCBunchSpacingNS = 10 * 1.e9 / LHCRFFreq;      // bunch spacing in ns (10 RFbuckets)
GPUglobalconstexpr() o2::gpu::GPUdoubleValue LHCOrbitNS = LHCMaxBunches * LHCBunchSpacingNS; // orbit duration in ns
GPUglobalconstexpr() o2::gpu::GPUdoubleValue LHCRevFreq = 1.e9 / LHCOrbitNS;                 // revolution frequency
GPUglobalconstexpr() o2::gpu::GPUdoubleValue LHCBunchSpacingMUS = LHCBunchSpacingNS * 1e-3;  // bunch spacing in \mus (10 RFbuckets)
GPUglobalconstexpr() o2::gpu::GPUdoubleValue LHCOrbitMUS = LHCOrbitNS * 1e-3;                // orbit duration in \mus
GPUglobalconstexpr() unsigned int MaxNOrbits = 0xffffffff;

// Offsets of A, C beam bunches at P2
GPUglobalconstexpr() int BunchOffsetsP2[2] = {344, 3017};

// convert LHC bunch ID to BC for 2 beam directions
constexpr int LHCBunch2P2BC(int bunch, BeamDirection dir)
{
  return (bunch + BunchOffsetsP2[int(dir)]) % LHCMaxBunches;
}

// convert BC at P2 to LHC bunch ID for 2 beam directions
constexpr int P2BC2LHCBunch(int bc, BeamDirection dir)
{
  int bunch = bc - BunchOffsetsP2[int(dir)];
  return bunch < 0 ? bunch + LHCMaxBunches : bunch;
}

} // namespace lhc
} // namespace constants
} // namespace o2

#endif

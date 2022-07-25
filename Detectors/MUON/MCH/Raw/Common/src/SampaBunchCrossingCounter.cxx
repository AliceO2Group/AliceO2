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

#include "MCHRawCommon/SampaBunchCrossingCounter.h"
#include "CommonConstants/LHCConstants.h"
#include "NofBits.h"
#include "MCHRawCommon/CoDecParam.h"

using namespace o2::constants::lhc;

namespace o2::mch::raw
{

constexpr int BCINORBIT = o2::constants::lhc::LHCMaxBunches;

uint20_t sampaBunchCrossingCounter(uint32_t orbit, uint16_t bc,
                                   uint32_t firstOrbit)
{
  orbit -= firstOrbit;
  auto bunchCrossingCounter = (orbit * LHCMaxBunches + bc) % ((1 << 20) - 1);
  impl::assertNofBits("bunchCrossingCounter", bunchCrossingCounter, 20);
  return bunchCrossingCounter;
}

std::tuple<uint32_t, uint16_t> orbitBC(uint20_t bunchCrossingCounter,
                                       uint32_t firstOrbit)
{
  impl::assertNofBits("bunchCrossingCounter", bunchCrossingCounter, 20);
  uint32_t orbit = bunchCrossingCounter / LHCMaxBunches + firstOrbit;
  int32_t bc = bunchCrossingCounter % LHCMaxBunches;
  return {orbit, bc};
}

} // namespace o2::mch::raw

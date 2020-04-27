// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFTDPLBASEPARAM_H_
#define ALICEO2_ITSMFTDPLBASEPARAM_H_

#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "CommonConstants/LHCConstants.h"
#include <string_view>

namespace o2
{
namespace itsmft
{

constexpr float DEFROFLength = o2::constants::lhc::LHCOrbitNS / 15;         // ROF length, divisor of the orbit
constexpr float DEFStrobeDelay = o2::constants::lhc::LHCBunchSpacingNS * 4; // ~100 ns delay

template <int N>
struct DPLAlpideParam : public o2::conf::ConfigurableParamHelper<DPLAlpideParam<N>> {
  static_assert(N == o2::detectors::DetID::ITS || N == o2::detectors::DetID::MFT, "only DetID::ITS orDetID:: MFT are allowed");

  static constexpr std::string_view getParamName()
  {
    return N == o2::detectors::DetID::ITS ? ParamName[0] : ParamName[1];
  }
  float roFrameLength = DEFROFLength;                 ///< length of RO frame in ns
  float strobeDelay = DEFStrobeDelay;                 ///< strobe start (in ns) wrt ROF start
  float strobeLength = DEFROFLength - DEFStrobeDelay; ///< length of the strobe in ns (sig. over threshold checked in this window only)

  // boilerplate stuff + make principal key
  O2ParamDef(DPLAlpideParam, getParamName().data());

 private:
  static constexpr std::string_view ParamName[2] = {"ITSAlpideParam", "MFTAlpideParam"};
};

template <int N>
DPLAlpideParam<N> DPLAlpideParam<N>::sInstance;

} // namespace itsmft
} // namespace o2

#endif

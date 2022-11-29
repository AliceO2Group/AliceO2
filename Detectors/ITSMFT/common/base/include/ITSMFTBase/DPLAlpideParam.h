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
constexpr float DEFStrobeDelay = o2::constants::lhc::LHCBunchSpacingNS * 4; // ~100 ns delay

template <int N>
struct DPLAlpideParam : public o2::conf::ConfigurableParamHelper<DPLAlpideParam<N>> {

  static constexpr std::string_view getParamName()
  {
    return N == o2::detectors::DetID::ITS ? ParamName[0] : ParamName[1];
  }
  int roFrameLengthInBC = DEFROFLengthBC();           ///< ROF length in BC for continuos mode
  float roFrameLengthTrig = DEFROFLengthTrig();       ///< length of RO frame in ns for triggered mode
  float strobeDelay = DEFStrobeDelay;                 ///< strobe start (in ns) wrt ROF start
  float strobeLengthCont = -1.;                       ///< if < 0, full ROF length - delay
  float strobeLengthTrig = 100.;                      ///< length of the strobe in ns (sig. over threshold checked in this window only)
  int roFrameBiasInBC = 0;                            ///< bias of the start of ROF wrt orbit start: t_irof = (irof*roFrameLengthInBC + roFrameBiasInBC)*BClengthMUS

  // boilerplate stuff + make principal key
  O2ParamDef(DPLAlpideParam, getParamName().data());

 private:
  static constexpr std::string_view ParamName[2] = {"ITSAlpideParam", "MFTAlpideParam"};

  static constexpr int DEFROFLengthBC()
  {
    // default ROF length in BC for continuos mode
    // allowed values: 1,2,3,4,6,9,11,12,18,22,27,33,36
    return N == o2::detectors::DetID::ITS ? o2::constants::lhc::LHCMaxBunches / 4 : o2::constants::lhc::LHCMaxBunches / 18;
  }
  static constexpr float DEFROFLengthTrig()
  {
    // length of RO frame in ns for triggered mode
    return N == o2::detectors::DetID::ITS ? 6000. : 6000.;
  }
  static_assert(N == o2::detectors::DetID::ITS || N == o2::detectors::DetID::MFT, "only DetID::ITS orDetID:: MFT are allowed");
  static_assert(o2::constants::lhc::LHCMaxBunches % DEFROFLengthBC() == 0); // make sure ROF length is divisor of the orbit
};

template <int N>
DPLAlpideParam<N> DPLAlpideParam<N>::sInstance;

} // namespace itsmft

namespace framework
{
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>> : std::true_type {
};
template <typename T>
struct is_messageable;
template <>
struct is_messageable<o2::itsmft::DPLAlpideParam<o2::detectors::DetID::MFT>> : std::true_type {
};

} // namespace framework

} // namespace o2

#endif

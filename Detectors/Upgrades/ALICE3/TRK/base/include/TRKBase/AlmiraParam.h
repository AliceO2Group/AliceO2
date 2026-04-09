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

#ifndef O2_TRK_ALMIRAPARAM_H
#define O2_TRK_ALMIRAPARAM_H

#include "CommonConstants/LHCConstants.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace trk
{
constexpr float DEFAlmiraStrobeDelay = 0.f; ///< default strobe delay in ns wrt ROF start, to be tuned with the real chip response

struct AlmiraParam : public o2::conf::ConfigurableParamHelper<AlmiraParam> {
  int roFrameLengthInBC = o2::constants::lhc::LHCMaxBunches / 198; ///< ROF length in BC for continuous mode
  float strobeDelay = DEFAlmiraStrobeDelay;                        ///< strobe start in ns wrt ROF start
  float strobeLengthCont = -1.;                                    ///< if < 0, full ROF length minus delay
  int roFrameBiasInBC = 0;                                         ///< ROF start bias in BC wrt orbit start

  O2ParamDef(AlmiraParam, "TRKAlmiraParam");
};

} // namespace trk

namespace framework
{
template <typename T>
struct is_messageable;

template <>
struct is_messageable<o2::trk::AlmiraParam> : std::true_type {
};
} // namespace framework

} // namespace o2

#endif

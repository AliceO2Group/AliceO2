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

#include <algorithm>

#include "CommonConstants/LHCConstants.h"
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "TRKBase/Specs.h"

namespace o2
{
namespace trk
{

struct AlmiraParam : public o2::conf::ConfigurableParamHelper<AlmiraParam> {
  static constexpr size_t kNLayers = constants::VD::petal::nLayers + constants::ML::nLayers + constants::OT::nLayers;
  static constexpr size_t getNLayers() { return kNLayers; }

  int roFrameLengthInBCPerLayer[kNLayers] = {0};  ///< ROF length in BC per layer
  float strobeDelayPerLayer[kNLayers] = {0};      ///< strobe delay in ns per layer
  float strobeLengthContPerLayer[kNLayers] = {0}; ///< strobe length in ns per layer
  int roFrameBiasInBCPerLayer[kNLayers] = {0};    ///< ROF start bias in BC per layer
  int roFrameDelayInBCPerLayer[kNLayers] = {0};   ///< extra ROF delay in BC per layer

  int getROFLengthInBC(int layer) const
  {
    if (roFrameLengthInBCPerLayer[layer] > 0) {
      return roFrameLengthInBCPerLayer[layer];
    } else {
      return o2::constants::lhc::LHCMaxBunches / 198;
    }
  }
  float getStrobeDelay(int layer) const { return strobeDelayPerLayer[layer]; }
  float getStrobeLengthCont(int layer) const { return strobeLengthContPerLayer[layer]; }
  int getROFBiasInBC(int layer) const { return roFrameBiasInBCPerLayer[layer]; }
  int getROFDelayInBC(int layer) const { return roFrameDelayInBCPerLayer[layer]; }

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

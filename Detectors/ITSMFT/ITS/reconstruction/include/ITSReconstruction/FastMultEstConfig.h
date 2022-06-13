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

/// \file  FastMultEstConfig.h
/// \brief Configuration parameters for ITS fast multiplicity estimator
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_ITS_FASTMULTESTCONF_H_
#define ALICEO2_ITS_FASTMULTESTCONF_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"
#include "ITSMFTReconstruction/ChipMappingITS.h"

namespace o2
{
namespace its
{
struct FastMultEstConfig : public o2::conf::ConfigurableParamHelper<FastMultEstConfig> {
  static constexpr int NLayers = o2::itsmft::ChipMappingITS::NLayers;

  /// acceptance correction per layer (relative to 1st one)
  float accCorr[NLayers] = {1.f, 0.895, 0.825, 0.803, 0.720, 0.962, 0.911};
  int firstLayer = 3;                            /// 1st layer to account
  int lastLayer = 6;                             /// last layer to account
  float imposeNoisePerChip = 1.e-7 * 1024 * 512; // assumed noise, free parameter if<0

  // cuts to reject to low or too high mult events
  float cutMultClusLow = 0;   /// reject ROF with estimated cluster mult. below this value (no cut if <0)
  float cutMultClusHigh = -1; /// reject ROF with estimated cluster mult. above this value (no cut if <0)
  float cutMultVtxLow = -1;   /// reject seed vertex if its multiplicity below this value (no cut if <0)
  float cutMultVtxHigh = -1;  /// reject seed vertex if its multiplicity above this value (no cut if <0)
  float cutRandomFraction = -1.; /// apply random cut rejecting requested fraction

  bool isMultCutRequested() const { return cutMultClusLow >= 0.f && cutMultClusHigh > 0.f; };
  bool isVtxMultCutRequested() const { return cutMultVtxLow >= 0.f && cutMultVtxHigh > 0.f; };
  bool isPassingRandomRejection() const;
  bool isPassingMultCut(float mult) const { return mult >= cutMultClusLow && (mult <= cutMultClusHigh || cutMultClusHigh <= 0.f); }
  bool isPassingVtxMultCut(int mult) const { return mult >= cutMultVtxLow && (mult <= cutMultVtxHigh || cutMultVtxHigh <= 0.f); }

  O2ParamDef(FastMultEstConfig, "fastMultConfig");
};

} // namespace its
} // namespace o2

#endif

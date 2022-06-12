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

#ifndef ALICEO2_ZDC_SIMCONFIG_H
#define ALICEO2_ZDC_SIMCONFIG_H

#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <array>

namespace o2
{
namespace zdc
{

struct ChannelSimCondition {
  using Histo = std::vector<float>;
  // Note: we don't use here correct BC spacing.
  // Shapes were measured with digitizer synchronized to a 100 MHz clock
  // As soon as we will have shapes acquired with the production digitizer we
  // will need to replace 25 with o2::constants::lhc::LHCBunchSpacingNS -> Done on 220609
  static constexpr float ShapeBinWidth = o2::constants::lhc::LHCBunchSpacingNS / NTimeBinsPerBC / 200;
  static constexpr float ShapeBinWidthInv = 1. / ShapeBinWidth;

  Histo shape;
  int ampMinID = 0;
  float pedestal = 0.f;
  float pedestalNoise = 0.f;
  float pedestalFluct = 0.f;
  float gain = 0.f;
  float gainInSum = 1.f;
  float timeJitter = 0.f;   // in ns
  float timePosition = 0.f; // in ns

  void print() const;
  ClassDefNV(ChannelSimCondition, 2);
};

struct SimCondition {

  std::array<ChannelSimCondition, NChannels> channels; // configuration per channel

  void print() const;

  ClassDefNV(SimCondition, 2);
};

}; // namespace zdc
} // namespace o2

#endif

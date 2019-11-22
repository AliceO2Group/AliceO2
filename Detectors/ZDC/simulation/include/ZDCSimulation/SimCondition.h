// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  static constexpr float ShapeBinWidth = 25. / NTimeBinsPerBC / 200; // Note: we don't use here correct BC spaces
  static constexpr float ShapeBinWidthInv = 1. / ShapeBinWidth;

  Histo shape;
  int ampMinID = 0;
  float pedestal = 0.f;
  float pedestalNoise = 0.f;
  float pedestalFluct = 0.f;
  float gain = 0.f;
  float timeJitter = 0.f;   // in ns
  float timePosition = 0.f; // in ns

  void print() const;
  ClassDefNV(ChannelSimCondition, 1);
};

struct SimCondition {

  std::array<ChannelSimCondition, NChannels> channels; // configuration per channel

  void print() const;

  ClassDefNV(SimCondition, 1);
};

}; // namespace zdc
} // namespace o2

#endif

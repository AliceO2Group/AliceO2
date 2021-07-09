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

#ifndef ZDC_ENERGY_H
#define ZDC_ENERGY_H

#include "ZDCBase/Constants.h"
#include <array>
#include <cmath>
#include <Rtypes.h>

/// \file ZDCEnergy.h
/// \brief Container class to store energy released in the ZDC
/// \author pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{

struct ZDCEnergy {

  uint32_t value = 0; // Signal id and energy released in calorimeter

  ZDCEnergy() = default;
  ZDCEnergy(uint8_t ch, float energy)
  {
    set(ch, energy);
  }
  inline void set(uint8_t ch, float energy)
  {
    float escaled = (energy + EnergyOffset) / EnergyUnit;
    value = 0;
    if (escaled > 0) {
      if (escaled > EnergyMask) {
        value = EnergyMask;
      } else {
        value = std::nearbyint(escaled);
      }
    }
    if (ch >= NChannels) {
      ch = 0x1f;
    }
    value = (value & EnergyMask) | (ch << 27);
  }
  float energy() const
  {
    return float(value & EnergyMask) * EnergyUnit - EnergyOffset;
  }
  uint8_t ch() const
  {
    return (value & EnergyChMask) >> 27;
  }

  void print() const;

  ClassDefNV(ZDCEnergy, 1);
};
} // namespace zdc
} // namespace o2

#endif

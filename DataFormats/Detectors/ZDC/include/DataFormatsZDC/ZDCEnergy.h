// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
/// \brief Container class to store energy released in a ZDC
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
  float energy()
  {
    return (float(value & EnergyMask) - EnergyOffset) * EnergyUnit;
  }
  uint8_t ch()
  {
    return (value & EnergyChMask) >> 27;
  }

  void print() const;

  ClassDefNV(ZDCEnergy, 1);
};
} // namespace zdc
} // namespace o2

#endif

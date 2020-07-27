// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/ElectronicsDelay.h
/// \brief  Delay parameters for MID electronics
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   27 July 2020
#ifndef O2_MID_ELECTRONICSDELAY_H
#define O2_MID_ELECTRONICSDELAY_H

#include <cstdint>
#include <iostream>

namespace o2
{
namespace mid
{

struct ElectronicsDelay {
  // The delays are in local clocks, and correspond to the LHC clocks (aka BCs)
  uint16_t calibToFET{9};
  uint16_t BCToLocal{0};
  uint16_t regToLocal{6};
};

std::ostream& operator<<(std::ostream& os, const ElectronicsDelay& delay);

ElectronicsDelay readElectronicsDelay(const char* filename);

} // namespace mid
} // namespace o2

#endif /* O2_MID_ELECTRONICSDELAY_H */

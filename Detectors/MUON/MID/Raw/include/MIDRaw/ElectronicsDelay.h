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

/// Electronics delays
///
/// The delays are in local clocks, and correspond to the LHC clocks (aka BCs)
struct ElectronicsDelay {
  uint16_t calibToFET{19}; ///< Delay between calibration and FET
  uint16_t BCToLocal{93};  ///< Delay between collision BC and local clock
  uint16_t regToLocal{6};  ///< Delay between regional board and local board answers
};

/// Output streamer for ElectronicsDelay
/// \param os Output stream
/// \param delay Electronics delay structure
std::ostream& operator<<(std::ostream& os, const ElectronicsDelay& delay);

/// Reads the electronic delays from file
///
/// The file must be in the form:
/// - keyword1 value1
/// - keyword2 value2
/// The available keywords are:
/// - calibToFET
/// - BCToLocal
/// - regToLocal
/// with the same meaning as the corresponding data member of the ElectronicsDelay structure.
/// If the keyword is not present in the file, the default value is used.
/// \param filename Path to file with delays
/// \return ElectronicDelay structure
ElectronicsDelay readElectronicsDelay(const char* filename);

} // namespace mid
} // namespace o2

#endif /* O2_MID_ELECTRONICSDELAY_H */

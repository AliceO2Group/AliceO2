// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/LocalBoardRO.h
/// \brief  Structure to store the FEE local board information
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   19 November 2019
#ifndef O2_MID_LocalBoardRO_H
#define O2_MID_LocalBoardRO_H

#include <cstdint>
#include <array>
#include <iosfwd>

namespace o2
{
namespace mid
{
struct LocalBoardRO {
  uint8_t statusWord{0};                 /// Status word
  uint8_t triggerWord{0};                /// Trigger word
  uint8_t boardId{0};                    /// Board ID in crate
  uint8_t firedChambers{0};              /// Fired chambers
  std::array<uint16_t, 4> patternsBP{};  /// Bending plane pattern
  std::array<uint16_t, 4> patternsNBP{}; /// Non-bending plane pattern
};

std::ostream& operator<<(std::ostream& os, const LocalBoardRO& loc);

namespace raw
{
static constexpr uint32_t sSTARTBIT = 1 << 7;
static constexpr uint32_t sCARDTYPE = 1 << 6;
static constexpr uint32_t sLOCALBUSY = 1 << 5;
static constexpr uint32_t sLOCALDECISION = 1 << 4;
static constexpr uint32_t sACTIVE = 1 << 3;
static constexpr uint32_t sREJECTING = 1 << 2;
static constexpr uint32_t sMASKED = 1 << 1;
static constexpr uint32_t sOVERWRITTEN = 1;

static constexpr uint32_t sSOX = 1 << 7;
static constexpr uint32_t sEOX = 1 << 6;
static constexpr uint32_t sPAUSE = 1 << 5;
static constexpr uint32_t sRESUME = 1 << 4;
static constexpr uint32_t sCALIBRATE = 1 << 3;
static constexpr uint32_t sPHY = 1 << 2;
static constexpr uint32_t sRESET = 1 << 1;
static constexpr uint32_t sORB = 1;

/// Tests the local card bit
inline bool isLoc(uint8_t statusWord) { return (statusWord >> 6) & 0x1; }
/// Tests the calibration bit of the card
inline bool isCalibration(uint8_t triggerWord) { return ((triggerWord & 0xc) == 0x8); }
/// Tests if this is a Front End Test event
inline bool isFET(uint8_t triggerWord) { return ((triggerWord & 0xc) == 0xc); }
} // namespace raw

} // namespace mid
} // namespace o2

#endif /* O2_MID_LocalBoardRO_H */

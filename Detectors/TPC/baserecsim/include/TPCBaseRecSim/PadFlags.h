// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @file   Defs.h
/// @author Jens Wiechula, Jens.Wiechula@ikf.uni-frankfurt.de
///

/// @brief  Global TPC definitions and constants

#ifndef AliceO2_TPC_PadFlags_H
#define AliceO2_TPC_PadFlags_H

namespace o2::tpc
{

enum class PadFlags : unsigned short {
  flagGoodPad = 1 << 0,      ///< flag for a good pad binary 0001
  flagDeadPad = 1 << 1,      ///< flag for a dead pad binary 0010
  flagUnknownPad = 1 << 2,   ///< flag for unknown status binary 0100
  flagSaturatedPad = 1 << 3, ///< flag for saturated status binary 0100
  flagHighPad = 1 << 4,      ///< flag for pad with extremly high IDC value
  flagLowPad = 1 << 5,       ///< flag for pad with extremly low IDC value
  flagSkip = 1 << 6,         ///< flag for defining a pad which is just ignored during the calculation of I1 and IDCDelta
  flagFEC = 1 << 7,          ///< flag for a whole masked FEC
  flagNeighbour = 1 << 8,    ///< flag if n neighbouring pads are outlier
  flagAllNoneGood = flagDeadPad | flagUnknownPad | flagSaturatedPad | flagHighPad | flagLowPad | flagSkip | flagFEC | flagNeighbour,
};

inline PadFlags operator&(PadFlags a, PadFlags b) { return static_cast<PadFlags>(static_cast<int>(a) & static_cast<int>(b)); }
inline PadFlags operator~(PadFlags a) { return static_cast<PadFlags>(~static_cast<int>(a)); }
inline PadFlags operator|(PadFlags a, PadFlags b) { return static_cast<PadFlags>(static_cast<int>(a) | static_cast<int>(b)); }

} // namespace o2::tpc

#endif

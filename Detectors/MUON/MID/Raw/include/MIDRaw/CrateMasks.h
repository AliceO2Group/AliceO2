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

/// \file   MIDRaw/CrateMasks.h
/// \brief  MID crate masks
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   11 March 2020
#ifndef O2_MID_CRATEMASKS_H
#define O2_MID_CRATEMASKS_H

#include <cstdint>
#include <array>
#include "CrateParameters.h"

namespace o2
{
namespace mid
{
class CrateMasks
{
 public:
  CrateMasks();
  CrateMasks(const char* filename);
  ~CrateMasks() = default;

  // Tests if the board in the feeId is active
  bool isActive(int iboard, uint8_t feeId) const { return mActiveBoards[feeId] & (1 << iboard); };

  /// Sets the active boards in the feeId
  void setActiveBoards(uint16_t feeId, uint8_t mask) { mActiveBoards[feeId] = mask; }

  /// Gets the mask for the feeId
  uint8_t getMask(uint16_t feeId) const { return mActiveBoards[feeId]; }

  void write(const char* filename) const;

 private:
  bool load(const char* filename);
  std::array<uint16_t, crateparams::sNGBTs> mActiveBoards; /// Active boards per GBT
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_CRATEMASKS_H */

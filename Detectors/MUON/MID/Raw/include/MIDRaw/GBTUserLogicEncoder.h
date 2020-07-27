// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/GBTUserLogicEncoder.h
/// \brief  Raw data encoder for MID GBT user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   20 April 2020
#ifndef O2_MID_GBTUSERLOGICENCODER_H
#define O2_MID_GBTUSERLOGICENCODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{
class GBTUserLogicEncoder
{
 public:
  void process(gsl::span<const LocalBoardRO> data, const uint16_t bc, uint8_t triggerWord = 0);
  void processTrigger(uint16_t bc, uint8_t triggerWord);

  /// Gets the buffer
  const std::vector<uint8_t>& getBuffer() { return mBytes; }

  /// Gets the buffer size in bytes
  size_t getBufferSize() const { return mBytes.size(); }

  /// Sets the mask
  void setMask(uint8_t mask) { mMask = mask; }

  /// Sets the feeID
  void setFeeId(uint16_t feeId) { mFeeId = feeId; }

  /// Clears the buffer
  void clear() { mBytes.clear(); }

  /// Sets the delay in the electronics
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mElectronicsDelay = electronicsDelay; }

 private:
  /// Adds the board id and the fired chambers
  inline void addIdAndChambers(uint8_t id, uint8_t firedChambers) { mBytes.emplace_back((id << 4) | firedChambers); }

  void addBoard(uint8_t statusWord, uint8_t triggerWord, uint16_t localClock, uint8_t id, uint8_t firedChambers);
  void addLoc(const LocalBoardRO& loc, uint16_t bc, uint8_t triggerWord);
  void addReg(uint16_t bc, uint8_t triggerWord, uint8_t id, uint8_t firedChambers);
  void addShort(uint16_t shortWord);
  bool checkAndAdd(gsl::span<const LocalBoardRO> data, uint16_t bc, uint8_t triggerWord);

  std::vector<uint8_t> mBytes{};      /// Vector with encoded information
  uint16_t mFeeId{0};                 /// FEE ID
  uint8_t mMask{0xFF};                /// GBT mask
  ElectronicsDelay mElectronicsDelay; /// Delays in the electronics
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTUSERLOGICENCODER_H */

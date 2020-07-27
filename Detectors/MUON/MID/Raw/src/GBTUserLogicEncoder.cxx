// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/GBTUserLogicEncoder.cxx
/// \brief  Raw data encoder for MID CRU user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019

#include "MIDRaw/GBTUserLogicEncoder.h"

#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

void GBTUserLogicEncoder::addShort(uint16_t shortWord)
{
  /// Adds a 16 bits word
  mBytes.emplace_back((shortWord >> 8) & 0xFF);
  mBytes.emplace_back(shortWord & 0xFF);
}

void GBTUserLogicEncoder::addBoard(uint8_t statusWord, uint8_t triggerWord, uint16_t localClock, uint8_t id, uint8_t firedChambers)
{
  /// Adds the board information
  mBytes.emplace_back(statusWord);
  mBytes.emplace_back(triggerWord);
  addShort(localClock);
  addIdAndChambers(id, firedChambers);
}

void GBTUserLogicEncoder::processTrigger(const uint16_t bc, uint8_t triggerWord)
{
  /// Adds the information in triggered mode
  for (int ireg = 0; ireg < 2; ++ireg) {
    uint8_t firedLoc = (mMask >> (4 * ireg)) & 0xF;
    addReg(bc, triggerWord, ireg, firedLoc);
  }
  for (int iloc = 0; iloc < 8; ++iloc) {
    if (mMask & (1 << iloc)) {
      addBoard(raw::sSTARTBIT | raw::sCARDTYPE, triggerWord, bc, iloc + 8 * crateparams::getGBTIdInCrate(mFeeId), 0);
      if (triggerWord & (raw::sSOX | raw::sEOX)) {
        /// Write masks
        for (int ich = 0; ich < 4; ++ich) {
          addShort(0xFFFF); // BP
          addShort(0xFFFF); // NBP
        }
      }
    }
  }
}

bool GBTUserLogicEncoder::checkAndAdd(gsl::span<const LocalBoardRO> data, const uint16_t bc, uint8_t triggerWord)
{
  /// Checks the local boards and write the regional and local output if needed
  uint8_t activeBoards = 0;
  for (auto& loc : data) {
    int regId = loc.boardId / 4;
    for (int ich = 0; ich < 4; ++ich) {
      if (loc.patternsBP[ich] && loc.patternsNBP[ich]) {
        activeBoards |= (1 << (loc.boardId % 8));
      }
    }
  }
  for (int ireg = 0; ireg < 2; ++ireg) {
    uint8_t firedLoc = (activeBoards >> (4 * ireg)) & 0xF;
    if (firedLoc > 0) {
      addReg(bc, triggerWord, ireg, firedLoc);
    }
  }
  for (auto& loc : data) {
    if (activeBoards & (1 << (loc.boardId % 8))) {
      addLoc(loc, bc, triggerWord);
    }
  }
  return (activeBoards > 0);
}

void GBTUserLogicEncoder::addReg(uint16_t bc, uint8_t triggerWord, uint8_t id, uint8_t firedChambers)
{
  /// Adds the regional board information
  mBytes.emplace_back(raw::sSTARTBIT);
  mBytes.emplace_back(triggerWord);
  uint16_t localClock = bc;
  if (triggerWord == 0) {
    localClock += mElectronicsDelay.BCToLocal + mElectronicsDelay.regToLocal;
  }
  addShort(localClock);
  addIdAndChambers(id + 8 * crateparams::getGBTIdInCrate(mFeeId), firedChambers);
}

void GBTUserLogicEncoder::addLoc(const LocalBoardRO& loc, uint16_t bc, uint8_t triggerWord)
{
  /// Adds the local board information

  mBytes.emplace_back(loc.statusWord);
  mBytes.emplace_back(triggerWord ? triggerWord : loc.triggerWord);

  uint16_t localClock = bc;
  if (loc.triggerWord == 0) {
    localClock += mElectronicsDelay.BCToLocal;
  }
  addShort(localClock);

  addIdAndChambers(loc.boardId, loc.firedChambers);
  for (int ich = 4; ich >= 0; --ich) {
    if (loc.firedChambers & (1 << ich)) {
      addShort(loc.patternsBP[ich]);
      addShort(loc.patternsNBP[ich]);
    }
  }
}

void GBTUserLogicEncoder::process(gsl::span<const LocalBoardRO> data, const uint16_t bc, uint8_t triggerWord)
{
  /// Encode data
  if (triggerWord != 0) {
    processTrigger(bc, triggerWord);
  }
  checkAndAdd(data, bc, triggerWord);
  if (triggerWord == raw::sCALIBRATE) {
    // Add FET
    checkAndAdd(data, bc + mElectronicsDelay.calibToFET, 0);
  }
}

} // namespace mid
} // namespace o2

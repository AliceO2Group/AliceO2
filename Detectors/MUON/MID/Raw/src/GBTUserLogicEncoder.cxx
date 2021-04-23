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

void GBTUserLogicEncoder::setGBTUniqueId(uint16_t gbtUniqueId)
{
  /// Sets the information associated to the GBT unique ID
  mCrateId = crateparams::getCrateIdFromGBTUniqueId(gbtUniqueId);
  mOffset = 8 * crateparams::getGBTIdInCrate(gbtUniqueId);
}

void GBTUserLogicEncoder::addShort(std::vector<char>& buffer, uint16_t shortWord) const
{
  /// Adds a 16 bits word
  buffer.emplace_back((shortWord >> 8) & 0xFF);
  buffer.emplace_back(shortWord & 0xFF);
}

void GBTUserLogicEncoder::processTrigger(const InteractionRecord& ir, uint8_t triggerWord)
{
  /// Adds the information in triggered mode
  auto& vec = mBoards[ir];
  for (uint8_t ireg = 0; ireg < 2; ++ireg) {
    uint8_t firedLoc = (mMask >> (4 * ireg)) & 0xF;
    vec.push_back({raw::sSTARTBIT, triggerWord, static_cast<uint8_t>(ireg + mOffset), firedLoc});
  }
  for (uint8_t iloc = 0; iloc < 8; ++iloc) {
    if (mMask & (1 << iloc)) {
      vec.push_back({raw::sSTARTBIT | raw::sCARDTYPE, triggerWord, static_cast<uint8_t>(iloc + mOffset), 0});
      if (triggerWord & (raw::sSOX | raw::sEOX)) {
        /// Write masks
        for (int ich = 0; ich < 4; ++ich) {
          vec.back().patternsBP[ich] = 0xFFFF;
          vec.back().patternsNBP[ich] = 0xFFFF;
        }
      }
    }
  }
}

void GBTUserLogicEncoder::addRegionalBoards(uint8_t activeBoards, InteractionRecord ir)
{
  /// Adds the regional board information
  ir += mElectronicsDelay.BCToLocal + mElectronicsDelay.regToLocal;
  auto& vec = mBoards[ir];
  for (uint8_t ireg = 0; ireg < 2; ++ireg) {
    uint8_t firedLoc = (activeBoards >> (4 * ireg)) & 0xF;
    if (firedLoc > 0) {
      vec.push_back({raw::sSTARTBIT, 0, static_cast<uint8_t>(ireg + mOffset), firedLoc});
    }
  }
}

void GBTUserLogicEncoder::process(gsl::span<const ROBoard> data, const InteractionRecord& ir)
{
  /// Encode data
  auto& vec = mBoards[ir];
  uint8_t activeBoards = 0;
  for (auto& loc : data) {
    for (int ich = 0; ich < 4; ++ich) {
      if (loc.patternsBP[ich] && loc.patternsNBP[ich]) {
        activeBoards |= (1 << (raw::getLocId(loc.boardId) % 8));
      }
    }
    vec.emplace_back(loc);
  }
  addRegionalBoards(activeBoards, ir);
}

void GBTUserLogicEncoder::flush(std::vector<char>& buffer, const InteractionRecord& ir)
{
  /// Flush buffer
  std::map<InteractionRecord, std::vector<ROBoard>> tmpBoards;
  for (auto& item : mBoards) {
    if (item.first <= ir) {
      for (auto& loc : item.second) {
        buffer.emplace_back(loc.statusWord);
        buffer.emplace_back(loc.triggerWord);
        addShort(buffer, item.first.bc);
        buffer.emplace_back((raw::getLocId(loc.boardId) << 4) | loc.firedChambers);
        buffer.emplace_back(mCrateId << 4);
        if (raw::isLoc(loc.statusWord)) {
          for (int ich = 4; ich >= 0; --ich) {
            if (loc.firedChambers & (1 << ich)) {
              addShort(buffer, loc.patternsBP[ich]);
              addShort(buffer, loc.patternsNBP[ich]);
            }
          }
        }
      }
    } else {
      tmpBoards[item.first] = item.second;
    }
  }
  mBoards.swap(tmpBoards);
}

} // namespace mid
} // namespace o2

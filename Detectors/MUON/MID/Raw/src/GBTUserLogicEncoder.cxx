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

void GBTUserLogicEncoder::setConfig(uint16_t gbtUniqueId, const std::vector<ROBoardConfig>& configurations)
{
  /// Sets the information associated to the GBT unique ID
  mCrateId = crateparams::getCrateIdFromGBTUniqueId(gbtUniqueId);
  mOffset = 8 * crateparams::getGBTIdInCrate(gbtUniqueId);
  mResponse.set(configurations);
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
  auto boards = mResponse.getTriggerResponse(triggerWord);
  vec.insert(vec.end(), boards.begin(), boards.end());
}

void GBTUserLogicEncoder::process(gsl::span<const ROBoard> data, InteractionRecord ir)
{
  /// Encode data
  ir += mElectronicsDelay.BCToLocal;

  // Apply zero suppression
  std::vector<ROBoard> zsLocs;
  for (auto& loc : data) {
    if (!mResponse.isZeroSuppressed(loc)) {
      zsLocs.emplace_back(loc);
    }
  }

  auto& vec = mBoards[ir];
  vec.insert(vec.end(), zsLocs.begin(), zsLocs.end());

  // Get regional response
  auto irReg = ir + mElectronicsDelay.regToLocal;
  auto regs = mResponse.getRegionalResponse(zsLocs);
  auto& vecReg = mBoards[irReg];
  vecReg.insert(vecReg.end(), regs.begin(), regs.end());
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

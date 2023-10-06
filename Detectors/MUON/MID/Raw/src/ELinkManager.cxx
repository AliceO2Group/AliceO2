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

/// \file   MID/Raw/src/ELinkManager.cxx
/// \brief  MID e-link data shaper manager
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 March 2021

#include "MIDRaw/ELinkManager.h"

#include "fmt/format.h"
#include "Framework/Logger.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{
void ELinkManager::init(uint16_t feeId, bool isDebugMode, bool isBare, const ElectronicsDelay& electronicsDelay, const FEEIdConfig& feeIdConfig)
{
  /// Initializer
  auto gbtUniqueIds = isBare ? std::vector<uint16_t>{feeId} : feeIdConfig.getGBTUniqueIdsInLink(feeId);
  mFeeId = feeId;

  for (auto& gbtUniqueId : gbtUniqueIds) {
    auto crateId = crateparams::getCrateIdFromGBTUniqueId(gbtUniqueId);
    uint8_t offset = crateparams::getGBTIdInCrate(gbtUniqueId) * 8;
    for (int ilink = 0; ilink < 10; ++ilink) {
      bool isLoc = ilink < 8;
      auto uniqueId = raw::makeUniqueLocID(crateId, ilink % 8 + offset);
      ELinkDataShaper shaper(isDebugMode, isLoc, uniqueId, electronicsDelay);
#if defined(MID_RAW_VECTORS)
      mDataShapers.emplace_back(shaper);
#else
      auto uniqueRegLocId = makeUniqueId(isLoc, uniqueId);
      mDataShapers.emplace(uniqueRegLocId, shaper);
#endif

      if (isBare) {
        ELinkDecoder decoder;
        decoder.setBareDecoder(true);
#if defined(MID_RAW_VECTORS)
        mDecoders.emplace_back(decoder);
#else
        mDecoders.emplace(uniqueRegLocId, decoder);
#endif
      }
    }
  }

#if defined(MID_RAW_VECTORS)
  if (isBare) {
    mIndex = [](uint8_t, uint8_t locId, bool isLoc) { return 8 * (1 - static_cast<size_t>(isLoc)) + (locId % 8); };
  } else {
    mIndex = [](uint8_t crateId, uint8_t locId, bool isLoc) { return 10 * (2 * (crateId % 4) + (locId / 8)) + 8 * (1 - static_cast<size_t>(isLoc)) + (locId % 8); };
  }
#endif
}

void ELinkManager::onDone(const ELinkDecoder& decoder, uint8_t crateId, uint8_t locId, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
{
  auto uniqueId = makeUniqueId(raw::isLoc(decoder.getStatusWord()), raw::makeUniqueLocID(crateId, locId));
  auto ds = mDataShapers.find(uniqueId);
  if (ds == mDataShapers.end()) {
    // There is something wrong: we are receiving data from a local board that is not expected to be there.
    // This usually happens when some local boards are not properly configured
    // and the crate or board ID that they return is not correct.
    // We keep track of this for debugging.
    // However, when a board is faulty it keeps sending messages.
    // To avoid flooding the logs, we warn only the first time we see it.
    auto& err = mErrors[uniqueId];
    ++err;
    static unsigned int nTotalErr = 0;
    if (err == 1 && nTotalErr++ < 3) {
      // This is the first time we see this faulty board, so we report it.
      ROBoard board{decoder.getStatusWord(), decoder.getTriggerWord(), raw::makeUniqueLocID(crateId, locId), decoder.getInputs()};
      for (int ich = 0; ich < 4; ++ich) {
        board.patternsBP[ich] = decoder.getPattern(0, ich);
        board.patternsNBP[ich] = decoder.getPattern(1, ich);
      }
      LOG(alarm) << "FEEId: " << mFeeId << "  board not found: " << board;
    }
    return;
  }
  return ds->second.onDone(decoder, data, rofs);
}

void ELinkManager::set(uint32_t orbit, uint32_t trigger)
{
  /// Setup the orbit
  for (auto& shaper : mDataShapers) {
#if defined(MID_RAW_VECTORS)
    shaper.set(orbit, trigger);
#else
    shaper.second.set(orbit, trigger);
#endif
  }
}

} // namespace mid
} // namespace o2

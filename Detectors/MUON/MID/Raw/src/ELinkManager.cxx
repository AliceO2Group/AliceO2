// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/ELinkManager.cxx
/// \brief  MID e-link data shaper manager
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 March 2021

#include "MIDRaw/ELinkManager.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{
void ELinkManager::init(uint16_t feeId, bool isDebugMode, bool isBare, const ElectronicsDelay& electronicsDelay, const FEEIdConfig& feeIdConfig)
{
  /// Initializer
  auto gbtUniqueIds = isBare ? std::vector<uint16_t>{feeId} : feeIdConfig.getGBTUniqueIdsInLink(feeId);

  for (auto& gbtUniqueId : gbtUniqueIds) {
    auto crateId = crateparams::getCrateIdFromGBTUniqueId(gbtUniqueId);
    uint8_t offset = crateparams::getGBTIdInCrate(gbtUniqueId) * 8;
    for (int ilink = 0; ilink < 10; ++ilink) {
      bool isLoc = ilink < 8;
      auto uniqueId = raw::makeUniqueLocID(crateId, ilink % 8 + offset);
      ELinkDataShaper shaper(isDebugMode, isLoc, uniqueId);
      shaper.setElectronicsDelay(electronicsDelay);
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

void ELinkManager::set(uint32_t orbit)
{
  /// Setup the orbit
  for (auto& shaper : mDataShapers) {
#if defined(MID_RAW_VECTORS)
    shaper.set(orbit);
#else
    shaper.second.set(orbit);
#endif
  }
}

} // namespace mid
} // namespace o2

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

/// \file   MIDRaw/ELinkManager.h
/// \brief  MID e-link data shaper manager
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 March 2021
#ifndef O2_MID_ELINKMANAGER_H
#define O2_MID_ELINKMANAGER_H

#include <cstdint>
#if defined(MID_RAW_VECTORS)
#include <vector>
#else
#include <unordered_map>
#endif
#include "MIDRaw/ELinkDataShaper.h"
#include "MIDRaw/ELinkDecoder.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"

namespace o2
{
namespace mid
{
class ELinkManager
{
 public:
  void init(uint16_t feeId, bool isDebugMode, bool isBare = false, const ElectronicsDelay& electronicsDelay = ElectronicsDelay(), const FEEIdConfig& feeIdConfig = FEEIdConfig());

  void set(uint32_t orbit, uint32_t trigger);

  /// Main function to be executed when decoding is done
  inline void onDone(const ELinkDecoder& decoder, uint8_t boardUniqueId, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs) { return onDone(decoder, raw::getCrateId(boardUniqueId), raw::getLocId(boardUniqueId), data, rofs); }

  /// Main function to be executed when decoding is done
  inline void onDone(const ELinkDecoder& decoder, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs) { return onDone(decoder, decoder.getCrateId(), decoder.getId(), data, rofs); }

#if defined(MID_RAW_VECTORS)
  /// Returns the decoder
  inline ELinkDecoder& getDecoder(uint8_t boardUniqueId, bool isLoc) { return mDecoders[mIndex(raw::getCrateId(boardUniqueId), raw::getLocId(boardUniqueId), isLoc)]; }

  /// Main function to be executed when decoding is done
  inline void onDone(const ELinkDecoder& decoder, uint8_t crateId, uint8_t locId, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs)
  {
    return mDataShapers[mIndex(crateId, locId, raw::isLoc(decoder.getStatusWord()))].onDone(decoder, data, rofs);
  }

 private:
  std::function<size_t(uint8_t, uint8_t, bool)> mIndex{}; ///! Function that returns the index in the vector
  std::vector<ELinkDataShaper> mDataShapers;              /// Vector with data shapers for each loc and reg board
  std::vector<ELinkDecoder> mDecoders;                    /// Vector with decoders for each loc and reg board

#else
  /// Returns the decoder
  inline ELinkDecoder& getDecoder(uint8_t boardUniqueId, bool isLoc) { return mDecoders.find(makeUniqueId(isLoc, boardUniqueId))->second; }

  /// Main function to be executed when decoding is done
  void onDone(const ELinkDecoder& decoder, uint8_t crateId, uint8_t locId, std::vector<ROBoard>& data, std::vector<ROFRecord>& rofs);

 private:
  /// Makes a ID which is unique for local and regional board
  inline uint16_t makeUniqueId(bool isLoc, uint8_t uniqueId) { return (isLoc ? 0 : (1 << 8)) | uniqueId; }
  std::unordered_map<uint16_t, ELinkDataShaper> mDataShapers; /// Data shapers for each loc and reg board
  std::unordered_map<uint16_t, ELinkDecoder> mDecoders;       /// Decoders for each loc and reg board
  std::unordered_map<uint16_t, unsigned long int> mErrors;    /// Decoding errors
  uint16_t mFeeId;                                            /// Front End ID

#endif
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ELINKMANAGER_H */

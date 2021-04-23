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
#include <map>
#include <gsl/gsl>
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "DataFormatsMID/ROBoard.h"

namespace o2
{
namespace mid
{
class GBTUserLogicEncoder
{
 public:
  void process(gsl::span<const ROBoard> data, const InteractionRecord& ir);
  void processTrigger(const InteractionRecord& ir, uint8_t triggerWord);

  void flush(std::vector<char>& buffer, const InteractionRecord& ir);

  // Encoder has no data left
  bool isEmpty() { return mBoards.empty(); }

  /// Sets the mask
  void setMask(uint8_t mask) { mMask = mask; }

  void setGBTUniqueId(uint16_t gbtUniqueId);

  /// Sets the delay in the electronics
  void setElectronicsDelay(const ElectronicsDelay& electronicsDelay) { mElectronicsDelay = electronicsDelay; }

 private:
  void addRegionalBoards(uint8_t activeBoards, InteractionRecord ir);
  void addShort(std::vector<char>& buffer, uint16_t shortWord) const;

  std::map<InteractionRecord, std::vector<ROBoard>> mBoards{}; /// Vector with boards
  uint8_t mCrateId{0};                                         /// Crate ID
  uint8_t mOffset{0};                                          /// GBT ID offset
  uint8_t mMask{0xFF};                                         /// GBT mask
  ElectronicsDelay mElectronicsDelay;                          /// Delays in the electronics
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_GBTUSERLOGICENCODER_H */

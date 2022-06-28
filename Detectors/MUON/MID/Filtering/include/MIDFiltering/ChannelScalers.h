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

/// \file   MIDFiltering/ChannelScalers.h
/// \brief  MID channel scalers
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2020
#ifndef O2_MID_CHANNELSCALERS_H
#define O2_MID_CHANNELSCALERS_H

#include <cstdint>
#include <unordered_map>
#include "DataFormatsMID/ColumnData.h"

namespace o2
{
namespace mid
{

class ChannelScalers
{
 public:
  /// Increments the counter per digit
  /// \param patterns List of noisy digits
  void count(const ColumnData& patterns);

  /// Resets scalers
  void reset() { mScalers.clear(); }

  /// Merges two counters
  void merge(const ChannelScalers& other);

  /// Gets deId from unique Id
  inline uint8_t getDeId(uint32_t uniqueId) const { return ((uniqueId >> 12) & 0x7F); }
  /// Gets columnId from unique Id
  inline uint8_t getColumnId(uint32_t uniqueId) const { return ((uniqueId >> 8) & 0xF); }
  /// Gets cathode from unique Id
  inline uint8_t getCathode(uint32_t uniqueId) const { return ((uniqueId >> 6) & 0x1); }
  /// Gets lineId from unique Id
  inline uint8_t getLineId(uint32_t uniqueId) const { return ((uniqueId >> 4) & 0x3); }
  /// Gets strip from unique Id
  inline uint8_t getStrip(uint32_t uniqueId) const { return (uniqueId & 0xF); }

  /// Gets the scalers
  const std::unordered_map<uint32_t, uint32_t>& getScalers() const { return mScalers; }

 private:
  /// Increments the counter for each digit
  /// \param deId Detection element ID
  /// \param columnId Column ID
  /// \param lineId Local board lin ID
  /// \param cathode Cathode or anode
  /// \param pattern Fired strip pattern
  void count(uint8_t deId, uint8_t columnId, int lineId, int cathode, uint16_t pattern);

  /// Returns a unique ID for the channel
  /// \param deId Detection element ID
  /// \param columnId Column ID
  /// \param lineId Local board lin ID
  /// \param strip Fired strip
  /// \param cathode Cathode or anode
  inline uint32_t getChannelId(uint8_t deId, uint8_t columnId, int lineId, int strip, int cathode) { return strip | (lineId << 4) | (cathode << 6) | (static_cast<uint32_t>(columnId) << 8) | (static_cast<uint32_t>(deId) << 12); }

  std::unordered_map<uint32_t, uint32_t> mScalers{}; // Channel scalers
};

/// Stream operator
std::ostream& operator<<(std::ostream& os, const ChannelScalers& channelScalers);

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHANNELSCALERS_H */

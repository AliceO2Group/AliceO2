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

/// \file   MIDFiltering/ChannelMasksHandler.h
/// \brief  MID channels masks handler
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 January 2020
#ifndef O2_MID_CHANNELMASKSHANDLER_H
#define O2_MID_CHANNELMASKSHANDLER_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "DataFormatsMID/ColumnData.h"

namespace o2
{
namespace mid
{

class ChannelMasksHandler
{
 public:
  void switchOffChannel(uint8_t deId, uint8_t columnId, int lineId, int strip, int cathode);
  void switchOffChannels(const ColumnData& dead);
  bool setFromChannelMask(const ColumnData& mask);
  bool setFromChannelMasks(const std::vector<ColumnData>& masks);
  bool applyMask(ColumnData& data) const;

  std::vector<ColumnData> getMasks() const;
  std::vector<ColumnData> getMasksFull(std::vector<ColumnData> referenceMask) const;

  /// Comparison operator
  bool operator==(const ChannelMasksHandler& right) const { return mMasks == right.mMasks; }

 private:
  ColumnData& getMask(uint8_t deId, uint8_t columnId);
  std::unordered_map<uint16_t, ColumnData> mMasks{}; // Channel masks
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHANNELMASKSHANDLER_H */

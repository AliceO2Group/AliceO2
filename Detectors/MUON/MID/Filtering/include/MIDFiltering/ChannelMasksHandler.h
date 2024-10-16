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
  /// Masks channel
  /// \param deId Detection element ID
  /// \param columnId Column ID
  /// \param lineId Local board line in the column
  /// \param strip Strip number
  /// \param cathode Anode or cathode
  void switchOffChannel(uint8_t deId, uint8_t columnId, int lineId, int strip, int cathode);

  /// Masks channels
  /// \param badChannels Bad channels
  void switchOffChannels(const ColumnData& badChannels);

  /// Masks channels
  /// \param badChannelsList List of bad channels
  void switchOffChannels(const std::vector<ColumnData>& badChannelsList);

  /// Sets the mask
  /// \param mask Mask to be added
  void setFromChannelMask(const ColumnData& mask);

  /// Sets the mask
  /// \param masks Masks
  void setFromChannelMasks(const std::vector<ColumnData>& masks);

  /// Applies the mask
  /// \param data Data to be masked. They will be modified
  /// \return false if the data is completely masked
  bool applyMask(ColumnData& data) const;

  /// Merges the masks
  /// \param masks Vector of masks to be merged
  void merge(const std::vector<ColumnData>& masks);

  /// Gets the masks
  std::vector<ColumnData> getMasks() const;

  /// Returns the masks map
  const std::unordered_map<uint16_t, ColumnData>& getMasksMap() const { return mMasks; }

  /// Comparison operator
  bool operator==(const ChannelMasksHandler& right) const { return mMasks == right.mMasks; }

  /// Clear masks
  void clear() { mMasks.clear(); }

 private:
  /// Gets the mask
  /// \param deId Detection element ID
  /// \param columnId Column ID
  /// \return Mask
  ColumnData& getMask(uint8_t deId, uint8_t columnId);

  std::unordered_map<uint16_t, ColumnData> mMasks{}; /// Channel masks
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_CHANNELMASKSHANDLER_H */

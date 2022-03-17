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

/// \file   MID/Filtering/src/ChannelMasksHandler.cxx
/// \brief  MID channels masks handler
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 January 2020

#include "MIDFiltering/ChannelMasksHandler.h"

namespace o2
{
namespace mid
{

ColumnData& ChannelMasksHandler::getMask(uint8_t deId, uint8_t columnId)
{
  auto uniqueId = getColumnDataUniqueId(deId, columnId);
  auto maskIt = mMasks.find(uniqueId);
  if (maskIt == mMasks.end()) {
    auto& newMask = mMasks[uniqueId];
    newMask.deId = deId;
    newMask.columnId = columnId;
    newMask.patterns.fill(0xFFFF);
    return newMask;
  }
  return maskIt->second;
}

void ChannelMasksHandler::switchOffChannel(uint8_t deId, uint8_t columnId, int lineId, int strip, int cathode)
{
  auto& mask = getMask(deId, columnId);
  uint16_t pattern = (1 << strip);
  if (cathode == 0) {
    mask.setBendPattern(mask.getBendPattern(lineId) & ~pattern, lineId);
  } else {
    mask.setNonBendPattern(mask.getNonBendPattern() & ~pattern);
  }
}

void ChannelMasksHandler::switchOffChannels(const ColumnData& badChannels)
{
  auto& mask = getMask(badChannels.deId, badChannels.columnId);
  mask.setNonBendPattern(mask.getNonBendPattern() & (~badChannels.getNonBendPattern()));
  for (int iline = 0; iline < 4; ++iline) {
    mask.setBendPattern(mask.getBendPattern(iline) & (~badChannels.getBendPattern(iline)), iline);
  }
}

void ChannelMasksHandler::switchOffChannels(const std::vector<ColumnData>& badChannelsList)
{
  for (auto& bad : badChannelsList) {
    switchOffChannels(bad);
  }
}

bool ChannelMasksHandler::applyMask(ColumnData& data) const
{
  auto uniqueId = getColumnDataUniqueId(data.deId, data.columnId);
  auto maskIt = mMasks.find(uniqueId);
  if (maskIt == mMasks.end()) {
    return true;
  }
  uint16_t allPatterns = 0;
  data.setNonBendPattern(data.getNonBendPattern() & maskIt->second.getNonBendPattern());
  allPatterns |= data.getNonBendPattern();
  for (int iline = 0; iline < 4; ++iline) {
    data.setBendPattern(data.getBendPattern(iline) & maskIt->second.getBendPattern(iline), iline);
    allPatterns |= data.getBendPattern(iline);
  }
  return (allPatterns != 0);
}

std::vector<ColumnData> ChannelMasksHandler::getMasks() const
{
  /// Gets the masks
  std::vector<ColumnData> masks;
  for (auto& maskIt : mMasks) {
    masks.emplace_back(maskIt.second);
  }
  return masks;
}

void ChannelMasksHandler::setFromChannelMask(const ColumnData& mask)
{
  /// Sets the mask from a channel mask
  auto uniqueColumnId = getColumnDataUniqueId(mask.deId, mask.columnId);
  mMasks[uniqueColumnId] = mask;
}

void ChannelMasksHandler::setFromChannelMasks(const std::vector<ColumnData>& masks)
{
  /// Sets the mask from a vector of channel masks
  mMasks.clear();
  for (auto& mask : masks) {
    setFromChannelMask(mask);
  }
}
void ChannelMasksHandler::merge(const std::vector<ColumnData>& masks)
{
  for (auto& mask : masks) {
    auto& inMask = getMask(mask.deId, mask.columnId);
    inMask.setNonBendPattern(inMask.getNonBendPattern() & mask.getNonBendPattern());
    for (int iline = 0; iline < 4; ++iline) {
      inMask.setBendPattern(inMask.getBendPattern(iline) & mask.getBendPattern(iline), iline);
    }
  }
}
} // namespace mid
} // namespace o2

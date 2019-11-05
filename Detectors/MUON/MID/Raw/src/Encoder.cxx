// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/Encoder.cxx
/// \brief  MID raw data fdata encoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/Encoder.h"

#include <map>
#include "RawInfo.h"

namespace o2
{
namespace mid
{
void Encoder::add(int value, unsigned int nBits)
{
  /// Add information
  for (int ibit = 0; ibit < nBits; ++ibit) {
    if (mBitIndex == raw::sElementSizeInBits) {
      mBitIndex = 0;
      if (mBytes.size() == mHeaderIndex + mHeaderOffset) {
        completePage(false);
        // Copy previous header
        std::vector<raw::RawUnit> oldHeader(mBytes.begin() + mHeaderIndex, mBytes.begin() + mHeaderIndex + raw::sHeaderSizeInElements);
        mHeaderIndex = mBytes.size();
        std::copy(oldHeader.begin(), oldHeader.end(), std::back_inserter(mBytes));
        // And increase the page counter
        ++getRDH()->pageCnt;
      }
      mBytes.push_back(0);
    }
    bool isOn = (value >> ibit) & 0x1;
    if (isOn) {
      mBytes.back() |= (1 << mBitIndex);
    }
    ++mBitIndex;
  }
}

void Encoder::clear()
{
  /// Reset bytes
  mBytes.clear();
  mBitIndex = 0;
  mHeaderIndex = 0;
}

void Encoder::newHeader(uint32_t bcId, uint32_t orbitId, uint32_t triggerType)
{
  /// Add new RDH
  completePage(true);
  mHeaderIndex = mBytes.size();
  for (size_t iel = 0; iel < raw::sHeaderSizeInElements; ++iel) {
    mBytes.emplace_back(0);
  }
  auto rdh = getRDH();
  *rdh = header::RAWDataHeader();
  rdh->triggerOrbit = orbitId;
  rdh->triggerBC = bcId;
  rdh->triggerType = triggerType;
  mBitIndex = raw::sElementSizeInBits;
}

void Encoder::completePage(bool stop)
{
  /// Complete the information on the page
  if (mBytes.empty()) {
    return;
  }
  auto pageSizeInElements = mBytes.size() - mHeaderIndex;
  if (mHeaderOffset > pageSizeInElements) {
    // Write zeros up to the end of the page
    mBytes.resize(mHeaderIndex + mHeaderOffset, 0);
  }

  auto rdh = getRDH();
  rdh->memorySize = pageSizeInElements * raw::sElementSizeInBytes;
  rdh->offsetToNext = (mBytes.size() - mHeaderIndex) * raw::sElementSizeInBytes;
  rdh->stop = stop;
}

const std::vector<raw::RawUnit>& Encoder::getBuffer()
{
  /// Gets the buffer
  completePage(true);
  return mBytes;
}

void Encoder::setHeaderOffset(uint16_t headerOffset)
{
  /// Sets the next header offset in bytes
  if (headerOffset < raw::sHeaderSizeInBytes) {
    // The offset must have at least the header size
    headerOffset = raw::sHeaderSizeInBytes;
  }
  mHeaderOffset = headerOffset / raw::sElementSizeInBytes;
}

void Encoder::process(gsl::span<const ColumnData> data, const uint16_t localClock, EventType eventType)
{
  /// Encode data
  std::map<int, std::vector<int>> dataIndexes;
  std::vector<std::pair<int, uint16_t>> patterns(4);
  for (size_t idx = 0; idx < data.size(); ++idx) {
    dataIndexes[data[idx].deId].emplace_back(idx);
  }
  add(static_cast<int>(eventType), sNBitsEventType);
  add(localClock, sNBitsLocalClock);
  add(dataIndexes.size(), sNBitsNFiredRPCs);
  for (auto& item : dataIndexes) {
    add(item.first, sNBitsRPCId);
    add(item.second.size(), sNBitsNFiredColumns);
    for (auto& icol : item.second) {
      patterns.clear();
      for (int iline = 0; iline < 4; ++iline) {
        if (data[icol].getBendPattern(iline) != 0) {
          patterns.emplace_back(iline, data[icol].getBendPattern(iline));
        }
      }
      if (patterns.empty()) {
        patterns.emplace_back(0, 0);
      }
      add(data[icol].columnId, sNBitsColumnId);
      add(patterns.size(), sNBitsNFiredBoards);
      for (auto& pat : patterns) {
        add(pat.first, sNBitsBoardId);
        add(data[icol].getNonBendPattern(), sNBitsFiredStrips);
        add(pat.second, sNBitsFiredStrips);
      }
    }
  }
}

} // namespace mid
} // namespace o2

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/Encoder.h
/// \brief  MID raw data encoder for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019
#ifndef O2_MID_ENCODER_H
#define O2_MID_ENCODER_H

#include <cstdint>
#include <vector>
#include <deque>
#include <gsl/gsl>
#include "Headers/RAWDataHeader.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/RawUnit.h"

namespace o2
{
namespace mid
{
class Encoder
{
 public:
  void newHeader(uint32_t bcId, uint32_t orbitId, uint32_t triggerType = 0);
  void process(gsl::span<const ColumnData> data, const uint16_t localClock, EventType eventType = EventType::Standard);
  const std::vector<raw::RawUnit>& getBuffer();
  /// Gets the buffer size in bytes
  size_t getBufferSize() const { return mBytes.size() * raw::sElementSizeInBytes; }
  /// Sets the next header offset in bytes
  void setHeaderOffset(uint16_t headerOffset = 0x2000);
  void clear();

 private:
  void add(int value, unsigned int nBits);
  void completePage(bool stop);
  header::RAWDataHeader* getRDH() { return reinterpret_cast<header::RAWDataHeader*>(&(mBytes[mHeaderIndex])); }

  std::vector<raw::RawUnit> mBytes{};                      /// Vector with encoded information
  size_t mBitIndex{0};                                     /// Index of the current bit
  size_t mHeaderIndex{0};                                  /// Index in the the current header
  size_t mHeaderOffset{0x2000 / raw::sElementSizeInBytes}; /// Header offset
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_ENCODER_H */

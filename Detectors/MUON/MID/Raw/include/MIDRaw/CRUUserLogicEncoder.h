// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/CRUUserLogicEncoder.h
/// \brief  Raw data encoder for MID CRU user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   18 November 2019
#ifndef O2_MID_CRUUSERLOGICENCODER_H
#define O2_MID_CRUUSERLOGICENCODER_H

#include <cstdint>
#include <vector>
#include <gsl/gsl>
#include "Headers/RAWDataHeader.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/LocalBoardRO.h"
#include "MIDRaw/RawUnit.h"

namespace o2
{
namespace mid
{
class CRUUserLogicEncoder
{
 public:
  void newHeader(uint16_t feeId, const header::RAWDataHeader& baseRDH);
  void process(gsl::span<const LocalBoardRO> data, const uint16_t bc, EventType eventType = EventType::Standard);
  const std::vector<raw::RawUnit>& getBuffer();
  /// Gets the buffer size in bytes
  size_t getBufferSize() const { return mBytes.size() * raw::sElementSizeInBytes; }
  /// Sets flag to add a constant header offset
  void setHeaderOffset(bool headerOffset = true) { mHeaderOffset = headerOffset; }
  void clear();

 private:
  void add(int value, unsigned int nBits);
  void completePage(bool stop);
  header::RAWDataHeader* getRDH() { return reinterpret_cast<header::RAWDataHeader*>(&(mBytes[mHeaderIndex])); }

  std::vector<raw::RawUnit> mBytes{}; /// Vector with encoded information
  size_t mBitIndex{0};                /// Index of the current bit
  size_t mHeaderIndex{0};             /// Index in the the current header
  bool mHeaderOffset{false};          /// Header offset
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRUUSERLOGICENCODER_H */

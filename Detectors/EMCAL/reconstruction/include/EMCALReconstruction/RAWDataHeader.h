// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef __O2_EMCAL_RAwDATAHEADER_H__
#define __O2_EMCAL_RAwDATAHEADER_H__

#include <iosfwd>
#include <cstdint>
#include "Rtypes.h"

namespace o2
{
namespace emcal
{
class RAWDataHeader
{
 public:
  RAWDataHeader() = default;
  ~RAWDataHeader() = default;

  uint16_t getEventID1() const { return (uint16_t)(mWord1 & 0xFFF); }
  uint16_t getBlockSize() const { return (uint16_t)((mBlockSizeOffset >> 16) & 0xFFFF); }
  uint16_t getOffset() const { return (uint16_t)(mBlockSizeOffset & 0xFFFF); }
  uint8_t getPacketCounter() const { return (uint8_t)((mPacketCounterLink >> 8) & 0xFF); }
  uint8_t getLink() const { return (uint8_t)(mPacketCounterLink & 0xFF); }
  uint8_t getL1TriggerMessage() const { return (uint8_t)((mWord1 >> 14) & 0xFF); }
  uint8_t getVersion() const { return (uint8_t)((mWord1 >> 24) & 0xFF); }
  uint32_t getStatus() const { return (mStatusMiniEventID >> 12) & 0xFFFF; }
  uint32_t getMiniEventID() const { return mStatusMiniEventID & 0xFFF; }
  uint64_t getTriggerClasses() const { return (((uint64_t)(mTriggerClassesMiddleLow & 0x3FFFF)) << 32) | mTriggerClassLow; }
  uint64_t getTriggerClassesNext50() const { return ((((uint64_t)(mROILowTriggerClassHigh & 0xF)) << 46) | ((uint64_t)mTriggerClassesMiddleHigh << 14) | (((uint64_t)mTriggerClassesMiddleLow >> 18) & 0x3fff)); }
  uint64_t getROI() const { return (((uint64_t)mROIHigh) << 4) | ((mROILowTriggerClassHigh >> 28) & 0xF); }
  void setTriggerClass(uint64_t mask)
  {
    mTriggerClassLow = (uint32_t)(mask & 0xFFFFFFFF);                                                          // low bits of trigger class
    mTriggerClassesMiddleLow = (mTriggerClassesMiddleLow & 0xFFFC0000) | ((uint32_t)((mask >> 32) & 0x3FFFF)); // middle low bits of trigger class
  };
  void SetTriggerClassNext50(uint64_t mask)
  {
    mTriggerClassesMiddleLow = (mTriggerClassesMiddleLow & 0x3FFFF) | (((uint32_t)(mask & 0x3FFF) << 18));
    mTriggerClassesMiddleHigh = (uint32_t)((mask >> 14) & 0xFFFFFFFF);                                 // middle high bits of trigger class
    mROILowTriggerClassHigh = (mROILowTriggerClassHigh & 0xFFFFFFF0) | (uint32_t)((mask >> 46) & 0xF); // low bits of ROI data (bits 28-31) and high bits of trigger class (bits 0-3)
  };

  void printStream(std::ostream& stream) const;
  void readStream(std::istream& stream);

 private:
  uint32_t mSize = 0xFFFFFFFF;            // size of the raw data in bytes
  uint32_t mWord1 = 3 << 24;              // bunch crossing, L1 trigger message and format version
  uint32_t mBlockSizeOffset = 0;          // Size [16-31] and offset [0-15]
  uint32_t mPacketCounterLink = 0;        // Number of packets [8-15] and linkID [0-7]
  uint32_t mStatusMiniEventID = 0x10000;  // status & error bits (bits 12-27) and mini event ID (bits 0-11)
  uint32_t mTriggerClassLow = 0;          // low bits of trigger class
  uint32_t mTriggerClassesMiddleLow = 0;  // 18 bits go into eventTriggerPattern[1] (low), 14 bits are zeroes (cdhMBZ2)
  uint32_t mTriggerClassesMiddleHigh = 0; // Goes into eventTriggerPattern[1] (high) and [2] (low)
  uint32_t mROILowTriggerClassHigh = 0;   // low bits of ROI data (bits 28-31) and high bits of trigger class (bits 0-17)
  uint32_t mROIHigh = 0;                  // high bits of ROI datadata */

  ClassDefNV(RAWDataHeader, 1);
};

std::istream& operator>>(std::istream& in, o2::emcal::RAWDataHeader& header);
std::ostream& operator<<(std::ostream& out, const o2::emcal::RAWDataHeader& header);

} // namespace emcal

} // namespace o2

#endif // _O2_EMCAL_RAwDATAHEADER_H__
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PixelData.h
/// \brief Transient data classes for single pixel and set of pixels from current chip
#ifndef ALICEO2_ITSMFT_PIXELDATA_H
#define ALICEO2_ITSMFT_PIXELDATA_H

#include "ITSMFTBase/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <vector>
#include <utility>
#include <cstdint>

namespace o2
{
namespace itsmft
{

///< single pixel datum, with possibility to set a flag of pixel being masked out
class PixelData
{

 public:
  PixelData(const Digit* dig) : mRow(dig->getRow()), mCol(dig->getColumn()) {}
  PixelData(uint16_t r = 0, uint16_t c = 0) : mRow(r), mCol(c) {}
  uint16_t getRow() const { return mRow & RowMask; }
  uint16_t getCol() const { return mCol; }
  bool isMasked() const { return mRow & MaskBit; }
  void setMask() { mRow |= MaskBit; }
  void unsetMask() { mRow &= RowMask; }

  /// for faster access when the pixel is guaranteed to not be masked
  uint16_t getRowDirect() const { return mRow; }

  bool operator==(const PixelData& dt) const
  {
    ///< check if one pixel is equal to another
    return (getCol() == dt.getCol()) && (getRow() == dt.getRow());
  }

  bool operator>(const PixelData& dt) const
  {
    ///< check if one pixel is greater than another (first column then row)
    if (getCol() == dt.getCol()) {
      return getRow() > dt.getRow();
    }
    return getCol() > dt.getCol();
  }

  bool operator<(const PixelData& dt) const
  {
    ///< check if one pixel is lesser than another (first column then row)
    if (getCol() == dt.getCol()) {
      return getRow() < dt.getRow();
    }
    return getCol() < dt.getCol();
  }

  int compare(const PixelData& dt) const
  {
    ///< compare to pixels (first column then row)
    return operator==(dt) ? 0 : (operator>(dt) ? 1 : -1);
  }

  static constexpr uint32_t DummyROF = 0xffffffff;
  static constexpr uint32_t DummyChipID = 0xffff;

 private:
  void sanityCheck() const;
  static constexpr int RowMask = 0x1ff; ///< 512 rows are supported
  static constexpr int MaskBit = 0x200; ///< 10-th bit is used to flag masked pixel
  uint16_t mRow = 0;                    ///< pixel row
  uint16_t mCol = 0;                    ///< pixel column

  ClassDefNV(PixelData, 1);
};

///< Transient data for single chip fired pixeld
///< Assumes that the digits data is sorted in chip/col/row
class ChipPixelData
{

 public:
  ChipPixelData() = default;
  ~ChipPixelData() = default;
  uint8_t getROFlags() const { return mROFlags; }
  uint16_t getChipID() const { return mChipID; }
  uint32_t getROFrame() const { return mROFrame; }
  uint32_t getStartID() const { return mStartID; }
  uint32_t getFirstUnmasked() const { return mFirstUnmasked; }
  uint32_t getTrigger() const { return mTrigger; }
  const o2::InteractionRecord& getInteractionRecord() const { return mInteractionRecord; }
  void setInteractionRecord(const o2::InteractionRecord& r) { mInteractionRecord = r; }
  const std::vector<PixelData>& getData() const { return mPixels; }
  std::vector<PixelData>& getData() { return (std::vector<PixelData>&)mPixels; }

  void setROFlags(uint8_t f = 0) { mROFlags = f; }
  void setChipID(uint16_t id) { mChipID = id; }
  void setROFrame(uint32_t r) { mROFrame = r; }
  void setStartID(uint32_t id) { mStartID = id; }
  void setFirstUnmasked(uint32_t n) { mFirstUnmasked = n; }
  void setTrigger(uint32_t t) { mTrigger = t; }

  void clear()
  {
    mPixels.clear();
    mROFlags = 0;
    mFirstUnmasked = 0;
  }

  void swap(ChipPixelData& other)
  {
    // swap content of two objects
    mPixels.swap(other.mPixels);
    std::swap(mROFrame, other.mROFrame);
    std::swap(mROFlags, other.mROFlags);
    std::swap(mTrigger, other.mTrigger);
    std::swap(mInteractionRecord, other.mInteractionRecord);
    std::swap(mChipID, other.mChipID);
    // strictly speaking, swapping the data below is not needed
    std::swap(mStartID, other.mStartID);
    std::swap(mFirstUnmasked, other.mFirstUnmasked);
  }

  void maskFiredInSample(const ChipPixelData& sample)
  {
    ///< mask in the current data pixels fired in provided sample
    const auto& pixelsS = sample.getData();
    uint32_t nC = mPixels.size();
    if (!nC) {
      return;
    }
    uint32_t nS = pixelsS.size();
    if (!nS) {
      return;
    }
    uint32_t itC = 0, itS = 0;
    while (itC < nC && itS < nS) {
      auto& pix0 = mPixels[itC];
      const auto& pixC = pixelsS[itS];
      if (pix0 == pixC) { // same
        pix0.setMask();
        if (mFirstUnmasked == itC++) { // mFirstUnmasked should flag 1st unmasked pixel entry
          mFirstUnmasked = itC;
        }
        itS++;
      } else if (pix0 < pixC) {
        itC++;
      } else {
        itS++;
      }
    }
  }

  void print() const;

 private:
  uint8_t mROFlags = 0;                          // readout flags from the chip trailer
  uint16_t mChipID = 0;                          // chip id within the detector
  uint32_t mROFrame = 0;                         // readout frame ID
  uint32_t mFirstUnmasked = 0;                   // first unmasked entry in the mPixels
  uint32_t mStartID = 0;                         // entry of the 1st pixel data in the whole detector data, for MCtruth access
  uint32_t mTrigger = 0;                         // trigger pattern
  o2::InteractionRecord mInteractionRecord = {}; // interaction record
  std::vector<PixelData> mPixels;                // vector of pixeld

  ClassDefNV(ChipPixelData, 1);
};
} // namespace itsmft
} // namespace o2

#endif //ALICEO2_ITSMFT_PIXELDATA_H

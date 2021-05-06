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

#include "DataFormatsITSMFT/Digit.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "ITSMFTReconstruction/DecodingStat.h"

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

  bool isNeighbour(const PixelData& dt, int maxDist) const
  {
    ///< check if one pixel is in proximity of another
    return (std::abs(static_cast<int>(getCol()) - static_cast<int>(dt.getCol())) <= maxDist &&
            std::abs(static_cast<int>(getRow()) - static_cast<int>(dt.getRow())) <= maxDist);
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

  void setError(ChipStat::DecErrors i) { mErrors |= 0x1 << i; }
  void setErrorFlags(uint32_t f) { mErrors |= f; }
  bool isErrorSet(ChipStat::DecErrors i) const { return mErrors & (0x1 << i); }
  bool isErrorSet() const { return mErrors != 0; }
  uint32_t getErrorFlags() const { return mErrors; }

  void clear()
  {
    mPixels.clear();
    mROFlags = 0;
    mFirstUnmasked = 0;
    mErrors = 0;
  }

  void swap(ChipPixelData& other)
  {
    // swap content of two objects
    std::swap(mROFlags, other.mROFlags);
    std::swap(mChipID, other.mChipID);
    std::swap(mROFrame, other.mROFrame);
    std::swap(mFirstUnmasked, other.mFirstUnmasked); // strictly speaking, not needed
    std::swap(mStartID, other.mStartID);             // strictly speaking, not needed
    std::swap(mTrigger, other.mTrigger);
    std::swap(mErrors, other.mErrors);
    std::swap(mInteractionRecord, other.mInteractionRecord);
    mPixels.swap(other.mPixels);
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

  void maskFiredInSample(const ChipPixelData& sample, int maxDist)
  {
    ///< mask in the current data pixels (or their neighbours) fired in provided sample
    const auto& pixelsS = sample.getData();
    int nC = mPixels.size();
    if (!nC) {
      return;
    }
    int nS = pixelsS.size();
    if (!nS) {
      return;
    }
    for (int itC = 0, itS = 0; itC < nC; itC++) {
      auto& pix0 = mPixels[itC];

      // seek to itS which is inferior than itC - maxDist
      auto mincol = pix0.getCol() > maxDist ? pix0.getCol() - maxDist : 0;
      auto minrow = pix0.getRowDirect() > maxDist ? pix0.getRowDirect() - maxDist : 0;
      if (itS == nS) { // in case itS lool below reached the end
        itS--;
      }
      while ((pixelsS[itS].getCol() > mincol || pixelsS[itS].getRow() > minrow) && itS > 0) {
        itS--;
      }
      for (; itS < nS; itS++) {
        const auto& pixC = pixelsS[itS];

        auto drow = static_cast<int>(pixC.getRow()) - static_cast<int>(pix0.getRowDirect());
        auto dcol = static_cast<int>(pixC.getCol()) - static_cast<int>(pix0.getCol());

        if (dcol > maxDist || (dcol == maxDist && drow > maxDist)) {
          break; // all higher itS will not match to this itC also
        }
        if (dcol < -maxDist || (drow > maxDist || drow < -maxDist)) {
          continue;
        } else {
          pix0.setMask();
          if (int(mFirstUnmasked) == itC) { // mFirstUnmasked should flag 1st unmasked pixel entry
            mFirstUnmasked = itC + 1;
          }
          break;
        }
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
  uint32_t mErrors = 0;                          // errors set during decoding
  o2::InteractionRecord mInteractionRecord = {}; // interaction record
  std::vector<PixelData> mPixels;                // vector of pixeld

  ClassDefNV(ChipPixelData, 1);
};
} // namespace itsmft
} // namespace o2

#endif //ALICEO2_ITSMFT_PIXELDATA_H

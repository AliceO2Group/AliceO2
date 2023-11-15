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
#ifndef ALICEO2_FOCAL_PIXELCHIP_H
#define ALICEO2_FOCAL_PIXELCHIP_H

#include <cstdint>
#include <iosfwd>
#include <vector>
#include "Rtypes.h"
#include <DataFormatsFOCAL/PixelHit.h>

namespace o2::focal
{

struct PixelChip {
  static constexpr uint16_t DATAFRAME = 0x1 << 0;
  static constexpr uint16_t EMPTYFRAME = 0x1 << 1;
  static constexpr uint16_t BUSY_ON = 0x1 << 2;
  static constexpr uint16_t BUSY_OFF = 0x1 << 3;

  uint8_t mFeeID;
  uint8_t mLaneID;
  uint8_t mChipID;
  uint16_t mStatusCode;
  std::vector<PixelHit> mHits;

  bool operator==(const PixelChip& other) const { return mChipID == other.mChipID && mLaneID == other.mLaneID && mFeeID == other.mFeeID; }
  bool operator<(const PixelChip& other) const
  {
    if (mFeeID < other.mFeeID) {
      return true;
    } else if (mFeeID == other.mFeeID) {
      if (mLaneID < other.mLaneID) {
        return true;
      } else if ((mLaneID == other.mLaneID) && (mChipID < other.mChipID)) {
        return true;
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  void setDataframe()
  {
    mStatusCode |= DATAFRAME;
  }

  void setEmptyframe()
  {
    mStatusCode |= EMPTYFRAME;
  }

  void setBusyOn()
  {
    mStatusCode |= BUSY_ON;
  }

  void removeDataframe()
  {
    mStatusCode &= ~(DATAFRAME);
  }

  void removeEmptyframe()
  {
    mStatusCode &= ~(EMPTYFRAME);
  }

  void removeBusyOn()
  {
    mStatusCode &= ~(BUSY_ON);
  }

  void removeBusyOff()
  {
    mStatusCode &= ~(BUSY_ON);
  }

  bool isDataframe() const
  {
    return mStatusCode & DATAFRAME;
  }

  bool isEmptyframe() const
  {
    return mStatusCode & EMPTYFRAME;
  }

  bool isBusyOn() const
  {
    return mStatusCode & BUSY_ON;
  }

  bool isBusyOff() const
  {
    return mStatusCode & BUSY_OFF;
  }

  ClassDefNV(PixelChip, 1);
};

std::ostream& operator<<(std::ostream& stream, const PixelChip& chip);
} // namespace o2::focal
#endif // QC_MODULE_FOCAL_PIXELCHIP_H
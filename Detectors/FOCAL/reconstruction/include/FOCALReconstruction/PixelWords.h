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
#ifndef ALICEO2_FOCAL_PIXELWORDS_H
#define ALICEO2_FOCAL_PIXELWORDS_H

#include <bitset>
#include <cstdint>

namespace o2::focal
{

namespace PixelWord
{

using Idle = uint8_t;
using BusyOn = uint8_t;
using BusyOff = uint8_t;

enum class PixelWordType {
  IDLE,
  CHIP_HEADER,
  CHIP_TRAILER,
  CHIP_EMPTYFRAME,
  REGION_HEADER,
  DATA_SHORT,
  DATA_LONG,
  BUSY_ON,
  BUSY_OFF,
  UNKNOWN
};

struct ChipHeader {
  union {
    struct {
      uint16_t mChipID : 4;
      uint16_t mIdentifier : 4;
      uint16_t mBunchCrossing : 8;
    };
    uint16_t mData;
  };

  bool isEmptyFrame() const { return mIdentifier == 0xe; }
};

struct ChipTrailer {
  union {
    struct {
      uint8_t mReadoutFlags : 4;
      uint8_t mIdentifier : 4;
    };
    uint8_t mData;
  };
};

struct RegionHeader {
  union {
    struct {
      uint8_t mRegion : 5;
      uint8_t mIdentifier : 3;
    };
    uint8_t mData;
  };
};

struct Hitmap {
  union {
    struct {
      uint8_t mHitmap : 7;
      uint8_t mZeroed : 1;
    };
    uint8_t mData;
  };

  std::bitset<7> getHitmap() const { return std::bitset<7>(mHitmap); }
};

struct DataShort {
  union {
    struct {
      uint16_t mAddress : 10;
      uint16_t mEncoderID : 4;
      uint16_t mIdentifier : 2;
    };
    uint16_t mData;
  };
  DataShort(const uint8_t* payload)
  {
    mData = (static_cast<uint16_t>(payload[0]) << 8) + static_cast<uint16_t>(payload[1]);
  }
  uint16_t getAddress() const { return mAddress; }
  uint16_t getEncoder() const { return mEncoderID; }
  uint16_t getIdentifier() const { return mIdentifier; }
};
} // namespace PixelWord

} // namespace o2::focal
#endif // ALICEO2_FOCAL_PIXELWORDS_H
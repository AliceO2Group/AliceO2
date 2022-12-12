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
  uint8_t mLaneID;
  uint8_t mChipID;
  std::vector<PixelHit> mHits;

  bool operator==(const PixelChip& other) const { return mChipID == other.mChipID && mLaneID == other.mLaneID; }
  bool operator<(const PixelChip& other) const
  {
    if (mLaneID < other.mLaneID) {
      return true;
    } else if ((mLaneID == other.mLaneID) && (mChipID < other.mChipID)) {
      return true;
    } else {
      return false;
    }
  }
  ClassDefNV(PixelChip, 1);
};

std::ostream& operator<<(std::ostream& stream, const PixelChip& chip);
} // namespace o2::focal
#endif // QC_MODULE_FOCAL_PIXELCHIP_H
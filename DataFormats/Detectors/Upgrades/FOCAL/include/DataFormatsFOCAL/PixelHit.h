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
#ifndef ALICEO2_FOCAL_PIXELHIT_H
#define ALICEO2_FOCAL_PIXELHIT_H

#include <cstdint>
#include <iosfwd>
#include "Rtypes.h"

namespace o2::focal
{

struct PixelHit {
  uint16_t mColumn;
  uint16_t mRow;

  bool operator==(const PixelHit& other) const { return mColumn == other.mColumn && mRow == other.mRow; }
  bool operator<(const PixelHit& other) const;

  ClassDefNV(PixelHit, 1);
};

std::ostream& operator<<(std::ostream& stream, const PixelHit& hit);

} // namespace o2::focal
#endif // ALICEO2_FOCAL_PIXELHIT_H
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

#include <iostream>
#include <DataFormatsFOCAL/PixelHit.h>

using namespace o2::focal;

bool PixelHit::operator<(const PixelHit& other) const
{
  if (mColumn < other.mColumn) {
    return true;
  } else if (mColumn == other.mColumn) {
    if (mRow < other.mRow) {
      return true;
    }
  }
  return false;
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PixelHit& hit)
{
  stream << "Pixel hit col " << hit.mColumn << ", row " << hit.mRow;
  return stream;
}

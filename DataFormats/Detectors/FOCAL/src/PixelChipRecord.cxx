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

#include <bitset>
#include <iostream>
#include "DataFormatsFOCAL/PixelChipRecord.h"

using namespace o2::focal;

void PixelChipRecord::printStream(std::ostream& stream) const
{
  stream << "Chip " << mChipID << " in layer " << mLayerID << " (lane " << mLaneID << "), number of hits: " << getNumberOfHits() << " (first hit: " << getFirstHit() << ")";
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const PixelChipRecord& chip)
{
  chip.printStream(stream);
  return stream;
}
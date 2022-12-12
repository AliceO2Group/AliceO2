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
#include "DataFormatsFOCAL/TriggerRecord.h"
#include "CommonConstants/Triggers.h"

using namespace o2::focal;

void TriggerRecord::printStream(std::ostream& stream) const
{
  stream << "Data for bc " << getBCData().bc << ", orbit " << getBCData().orbit
         << ", [Pixels Chips] starting from entry " << getFirstPixelChipEntry() << " with " << getNumberOfPixelChipObjects()
         << ", [Pixels Hits] starting from entry " << getFirstPixelHitEntry() << " with " << getNumberOfPixelHitObjects()
         << " objects, [Pads] starting from entry " << getFirstPadEntry() << " with " << getNumberOfPadObjects();
}

std::ostream& o2::focal::operator<<(std::ostream& stream, const TriggerRecord& trg)
{
  trg.printStream(stream);
  return stream;
}
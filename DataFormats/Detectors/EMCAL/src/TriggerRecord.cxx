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
#include "DataFormatsEMCAL/TriggerRecord.h"

namespace o2
{

namespace emcal
{

void TriggerRecord::printStream(std::ostream& stream) const
{
  stream << "Data for bc " << getBCData().bc << ", orbit " << getBCData().orbit << ", Triggers " << std::bitset<sizeof(mTriggerBits) * 8>(mTriggerBits) << ", starting from entry " << getFirstEntry() << " with " << getNumberOfObjects() << " objects";
}

std::ostream& operator<<(std::ostream& stream, const TriggerRecord& trg)
{
  trg.printStream(stream);
  return stream;
}
} // namespace emcal
} // namespace o2
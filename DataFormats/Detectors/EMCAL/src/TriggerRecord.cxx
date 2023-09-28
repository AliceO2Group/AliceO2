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
#include "CommonConstants/Triggers.h"
#include "DataFormatsEMCAL/Constants.h"

namespace o2
{

namespace emcal
{

uint16_t TriggerRecord::getTriggerBitsCompressed() const
{
  uint16_t result(0);
  if (mTriggerBits & o2::trigger::PhT) {
    result |= 1 << TriggerBitsCoded_t::PHYSTRIGGER;
  }
  if (mTriggerBits & o2::trigger::Cal) {
    result |= 1 << TriggerBitsCoded_t::CALIBTRIGGER;
  }
  if (mTriggerBits & o2::emcal::triggerbits::Inc) {
    result |= 1 << TriggerBitsCoded_t::REJECTINCOMPLETE;
  }
  return result;
}

void TriggerRecord::setTriggerBitsCompressed(uint16_t triggerbits)
{
  mTriggerBits = 0;
  if (triggerbits & (1 << TriggerBitsCoded_t::PHYSTRIGGER)) {
    mTriggerBits |= o2::trigger::PhT;
  }
  if (triggerbits & (1 << TriggerBitsCoded_t::CALIBTRIGGER)) {
    mTriggerBits |= o2::trigger::Cal;
  }
  if (triggerbits & (1 << TriggerBitsCoded_t::REJECTINCOMPLETE)) {
    mTriggerBits |= o2::emcal::triggerbits::Inc;
  }
}

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
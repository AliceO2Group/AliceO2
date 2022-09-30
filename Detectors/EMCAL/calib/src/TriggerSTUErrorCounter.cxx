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

#include "EMCALCalib/TriggerSTUErrorCounter.h"

#include <fairlogger/Logger.h>

#include <bitset>
#include <iomanip>

using namespace o2::emcal;

TriggerSTUErrorCounter::TriggerSTUErrorCounter(int Time, unsigned long Error)
{
  setValue(Time, Error);
}

TriggerSTUErrorCounter::TriggerSTUErrorCounter(std::pair<int, unsigned long> TimeAndError)
{
  setValue(TimeAndError);
}

bool TriggerSTUErrorCounter::operator==(const TriggerSTUErrorCounter& other) const
{
  return (mTimeErrorCount == other.mTimeErrorCount);
}

bool TriggerSTUErrorCounter::isEqual(TriggerSTUErrorCounter& counter) const
{
  return counter.mTimeErrorCount.first == mTimeErrorCount.first;
}

int TriggerSTUErrorCounter::compare(TriggerSTUErrorCounter& counter) const
{
  if (mTimeErrorCount.first > counter.mTimeErrorCount.first) {
    return 1;
  }
  if (mTimeErrorCount.first < counter.mTimeErrorCount.first) {
    return -1;
  }

  return 0;
}

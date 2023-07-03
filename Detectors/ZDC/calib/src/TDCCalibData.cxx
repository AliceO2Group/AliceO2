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

#include "Framework/Logger.h"
#include "ZDCCalib/TDCCalibData.h"

using namespace o2::zdc;

void TDCCalibData::print() const
{
  for (int i = 0; i < NTDC; i++) {
    LOGF(info, "%s entries: %d", CTDC[i], entries[i]);
  }
}

TDCCalibData& TDCCalibData::operator+=(const TDCCalibData& other)
{

  for (int32_t ih = 0; ih < NTDC; ih++) {
    entries[ih] = entries[ih] + other.entries[ih];
  }

  if (mCTimeBeg == 0 || other.mCTimeBeg < mCTimeBeg) {
    mCTimeBeg = other.mCTimeBeg;
  }
  if (other.mCTimeEnd > mCTimeEnd) {
    mCTimeEnd = other.mCTimeEnd;
  }
#ifdef O2_ZDC_DEBUG
  LOGF(info, "TDCCalibData [%llu : %llu]: %s=%d %s=%d %s=%d %s=%d %s=%d %s=%d %s=%d %s=%d %s=%d %s=%d", mCTimeBeg, mCTimeEnd, CTDC[0], getEntries(0), CTDC[1], getEntries(1),
       CTDC[2], getEntries(2), CTDC[3], getEntries(3), CTDC[4], getEntries(4), CTDC[5], getEntries(5), CTDC[6], getEntries(6), CTDC[7], getEntries(7),
       CTDC[8], getEntries(8), CTDC[9], getEntries(9));
#endif
  return *this;
}

void TDCCalibData::setCreationTime(uint64_t ctime)
{
  mCTimeBeg = ctime;
  mCTimeEnd = ctime;
#ifdef O2_ZDC_DEBUG
  LOGF(info, "TDCCalibData::setCreationTime %llu", ctime);
#endif
}

int TDCCalibData::getEntries(int ih) const
{
  if (ih < 0 || ih >= NTDC) {
    LOGF(error, "TDCCalibData::getEntries ih = %d is out of range", ih);
    return 0;
  }
  return entries[ih];
}

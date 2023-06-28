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
#include "ZDCCalib/InterCalibData.h"

using namespace o2::zdc;

void InterCalibData::print() const
{
  for (int i = 0; i < NH; i++) {
    LOGF(info, "%s", DN[i]);
    for (int j = 0; j < NPAR; j++) {
      for (int k = 0; k < NPAR; k++) {
        if (k == 0) {
          printf("%e", mSum[i][j][k]);
        } else {
          printf(" %e", mSum[i][j][k]);
        }
      }
      printf("\n");
    }
  }
}

InterCalibData& InterCalibData::operator+=(const InterCalibData& other)
{
  for (int32_t ih = 0; ih < NH; ih++) {
    for (int32_t i = 0; i < NPAR; i++) {
      for (int32_t j = 0; j < NPAR; j++) {
        mSum[ih][i][j] += other.mSum[ih][i][j];
      }
    }
  }
  if (mCTimeBeg == 0 || other.mCTimeBeg < mCTimeBeg) {
    mCTimeBeg = other.mCTimeBeg;
  }
  if (other.mCTimeEnd > mCTimeEnd) {
    mCTimeEnd = other.mCTimeEnd;
  }
#ifdef O2_ZDC_DEBUG
  LOGF(info, "InterCalibData [%llu : %llu]: %s=%d %s=%d %s=%d %s=%d %s=%d", mCTimeBeg, mCTimeEnd, DN[0], getEntries(0), DN[1], getEntries(1),
       DN[2], getEntries(2), DN[3], getEntries(3), DN[4], getEntries(4));
#endif
  return *this;
}

void InterCalibData::setCreationTime(uint64_t ctime)
{
  mCTimeBeg = ctime;
  mCTimeEnd = ctime;
#ifdef O2_ZDC_DEBUG
  LOGF(info, "InterCalibData::setCreationTime %llu", ctime);
#endif
}

int InterCalibData::getEntries(int ih) const
{
  if (ih < 0 || ih >= NH) {
    LOGF(error, "InterCalibData::getEntries ih = %d is out of range", ih);
    return 0;
  }
  return mSum[ih][5][5];
}

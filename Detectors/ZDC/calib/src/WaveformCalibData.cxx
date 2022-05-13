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
#include "ZDCCalib/WaveformCalibData.h"

using namespace o2::zdc;

void WaveformCalibData::print() const
{
  for (int i = 0; i < NH; i++) {
    LOGF(info, "%d ", i);
  }
}

WaveformCalibData& WaveformCalibData::operator+=(const WaveformCalibData& other)
{
//   for (int32_t ih = 0; ih < NH; ih++) {
//     for (int32_t i = 0; i < NPAR; i++) {
//       for (int32_t j = 0; j < NPAR; j++) {
//         mSum[ih][i][j] += other.mSum[ih][i][j];
//       }
//     }
//   }
//   if (mCTimeBeg == 0 || other.mCTimeBeg < mCTimeBeg) {
//     mCTimeBeg = other.mCTimeBeg;
//   }
//   if (other.mCTimeEnd > mCTimeEnd) {
//     mCTimeEnd = other.mCTimeEnd;
//   }
//   //#ifdef O2_ZDC_DEBUG
//   LOGF(info, "WaveformCalibData [%llu : %llu]: %s=%d %s=%d %s=%d %s=%d %s=%d", mCTimeBeg, mCTimeEnd, DN[0], getEntries(0), DN[1], getEntries(1),
//        DN[2], getEntries(2), DN[3], getEntries(3), DN[4], getEntries(4));
//   //#endif
  return *this;
}

void WaveformCalibData::setCreationTime(uint64_t ctime)
{
  mCTimeBeg = ctime;
  mCTimeEnd = ctime;
#ifdef O2_ZDC_DEBUG
  LOGF(info, "WaveformCalibData::setCreationTime %llu", ctime);
#endif
}

int WaveformCalibData::getEntries(int ih) const
{
  if (ih < 0 || ih >= NH) {
    LOGF(error, "WaveformCalibData::getEntries ih = %d is out of range", ih);
    return 0;
  }
  return 0; // TODO
}

void WaveformCalibData::setN(int n)
{
  if (n >= 0 && n < WaveformCalibConfig::NBT) {
    mN = n;
    for (int ih = 0; ih < NH; ih++) {
      mFirstValid[ih] = 0;
      mLastValid[ih] = NH * NTimeBinsPerBC * TSN - 1;
    }
  } else {
    LOG(fatal) << "WaveformCalibData " << __func__ << " wrong stored b.c. setting " << n << " not in range [0:" << WaveformCalibConfig::NBT << "]";
  }
}

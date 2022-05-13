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
  for (int32_t ih = 0; ih < NH; ih++) {
    LOGF(info, "WaveformCalibData [%llu : %llu]: entries=%d [%d:%d:%d]", mCTimeBeg, mCTimeEnd, mEntries[ih], mFirstValid[ih], mPeak, mLastValid[ih]);
  }
}

WaveformCalibData& WaveformCalibData::operator+=(const WaveformCalibData& other)
{
  if (mN != other.mN) {
    LOG(fatal) << "Mixing waveform with different configurations mN = " << mN << " != " << other.mN;
    return *this;
  }
  if (mPeak != other.mPeak) {
    LOG(fatal) << "Mixing waveform with different configurations mPeak = " << mPeak << " != " << other.mPeak;
    return *this;
  }
  if (mCTimeBeg == 0 || other.mCTimeBeg < mCTimeBeg) {
    mCTimeBeg = other.mCTimeBeg;
  }
  if (other.mCTimeEnd > mCTimeEnd) {
    mCTimeEnd = other.mCTimeEnd;
  }
  for (int32_t ih = 0; ih < NH; ih++) {
    if (mFirstValid[ih] > other.mFirstValid[ih]) {
      mFirstValid[ih] = other.mFirstValid[ih];
    }
    if (mLastValid[ih] > other.mLastValid[ih]) {
      mLastValid[ih] = other.mLastValid[ih];
    }
    mEntries[ih] = mEntries[ih] + other.mEntries[ih];
    for (int32_t i = mFirstValid[ih]; i <= mLastValid[ih]; i++) {
      mWave[ih][i] = mWave[ih][i] + other.mWave[ih][i];
    }
  }
  //#ifdef O2_ZDC_DEBUG
  print();
  //#endif
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

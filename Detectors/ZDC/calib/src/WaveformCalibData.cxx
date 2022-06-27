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

#include <TH1.h>
#include <TFile.h>
#include <TDirectory.h>
#include "Framework/Logger.h"
#include "ZDCCalib/WaveformCalibData.h"

using namespace o2::zdc;

//______________________________________________________________________________
void WaveformCalibData::print() const
{
  LOGF(info, "WaveformCalibData mN = %d/%d [%llu : %llu]", mN, NBT, mCTimeBeg, mCTimeEnd);
  for (int32_t is = 0; is < NChannels; is++) {
    if (getEntries(is) > 0) {
      LOGF(info, "WaveformCalibData %2d %s: entries=%d [%d:%d:%d]", is, ChannelNames[is].data(), getEntries(is), getFirstValid(is), mPeak, getLastValid(is));
    }
  }
}

//______________________________________________________________________________
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
  for (int32_t is = 0; is < NChannels; is++) {
    mWave[is] += other.mWave[is];
  }
#ifdef O2_ZDC_DEBUG
  LOG(info) << __func__;
  print();
#endif
  return *this;
}

WaveformCalibChData& WaveformCalibChData::operator+=(const WaveformCalibChData& other)
{
  if (other.mEntries > 0) {
    if (other.mFirstValid > mFirstValid) {
      mFirstValid = other.mFirstValid;
    }
    if (other.mLastValid < mLastValid) {
      mLastValid = other.mLastValid;
    }
    mEntries = mEntries + other.mEntries;
    for (int32_t i = mFirstValid; i <= mLastValid; i++) {
      mData[i] = mData[i] + other.mData[i];
    }
  }
  return *this;
}

//______________________________________________________________________________
void WaveformCalibData::setCreationTime(uint64_t ctime)
{
  mCTimeBeg = ctime;
  mCTimeEnd = ctime;
#ifdef O2_ZDC_DEBUG
  LOGF(info, "WaveformCalibData::setCreationTime %llu", ctime);
#endif
}

//______________________________________________________________________________
int WaveformCalibData::getEntries(int is) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "WaveformCalibData::getEntries channel index %d is out of range", is);
    return 0;
  }
  return mWave[is].getEntries();
}

int WaveformCalibChData::getEntries() const
{
  return mEntries;
}

//______________________________________________________________________________
int WaveformCalibData::getFirstValid(int is) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "WaveformCalibData::getFirstValid channel index %d is out of range", is);
    return 0;
  }
  return mWave[is].getFirstValid();
}

int WaveformCalibChData::getFirstValid() const
{
  return mFirstValid;
}

//______________________________________________________________________________
int WaveformCalibData::getLastValid(int is) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "WaveformCalibData::getLastValid channel index %d is out of range", is);
    return 0;
  }
  return mWave[is].getLastValid();
}

int WaveformCalibChData::getLastValid() const
{
  return mLastValid;
}

//______________________________________________________________________________
void WaveformCalibData::setN(int n)
{
  if (n >= 0 && n < NBT) {
    mN = n;
    for (int is = 0; is < NChannels; is++) {
      mWave[is].setN(n);
    }
  } else {
    LOG(fatal) << "WaveformCalibData " << __func__ << " wrong stored b.c. setting " << n << " not in range [0:" << NBT << "]";
  }
}

void WaveformCalibChData::setN(int n)
{
  if (n >= 0 && n < NBT) {
    mFirstValid = 0;
    mLastValid = n * NTimeBinsPerBC * TSN - 1;
  } else {
    LOG(fatal) << "WaveformCalibChData " << __func__ << " wrong stored b.c. setting " << n << " not in range [0:" << NBT << "]";
  }
}

//______________________________________________________________________________
int WaveformCalibData::saveDebugHistos(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    if (mWave[is].mEntries > 0) {
      TString n = TString::Format("h%d", is);
      TString t = TString::Format("Waveform %d %s", is, ChannelNames[is].data());
      int nbx = mWave[is].mLastValid - mWave[is].mFirstValid + 1;
      float min = mWave[is].mFirstValid - mPeak - 0.5;
      float max = mWave[is].mLastValid - mPeak + 0.5;
      TH1F h(n, t, nbx, min, max);
      for (int ibx = 0; ibx < nbx; ibx++) {
        h.SetBinContent(ibx + 1, mWave[is].mData[mWave[is].mFirstValid + ibx]);
      }
      h.SetEntries(mWave[is].mEntries);
      h.Write("", TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}

//______________________________________________________________________________
void WaveformCalibData::clear()
{
  mCTimeBeg = 0;
  mCTimeEnd = 0;
  mN = 0;
  mPeak = 0;
  for (int32_t is = 0; is < NChannels; is++) {
    mWave[is].clear();
  }
}

void WaveformCalibChData::clear()
{
  mEntries = 0;
  mFirstValid = -1;
  mLastValid = -1;
  for (int iw = 0; iw < NW; iw++) {
    mData[iw] = 0;
  }
}

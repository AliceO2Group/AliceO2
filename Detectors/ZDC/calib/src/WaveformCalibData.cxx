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

void WaveformCalibData::print() const
{
  LOGF(info, "WaveformCalibData mN = %d/%d", mN, NBT);
  for (int32_t is = 0; is < NChannels; is++) {
    if (mEntries[is] > 0) {
      LOGF(info, "WaveformCalibData %2d %s [%llu : %llu]: entries=%d [%d:%d:%d]", is, ChannelNames[is].data(), mCTimeBeg, mCTimeEnd, mEntries[is], mFirstValid[is], mPeak, mLastValid[is]);
    }
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
  for (int32_t is = 0; is < NChannels; is++) {
    if (other.mEntries[is] > 0) {
      if (other.mFirstValid[is] > mFirstValid[is]) {
        mFirstValid[is] = other.mFirstValid[is];
      }
      if (other.mLastValid[is] < mLastValid[is]) {
        mLastValid[is] = other.mLastValid[is];
      }
      mEntries[is] = mEntries[is] + other.mEntries[is];
      for (int32_t i = mFirstValid[is]; i <= mLastValid[is]; i++) {
        mWave[is][i] = mWave[is][i] + other.mWave[is][i];
      }
    }
  }
#ifdef O2_ZDC_DEBUG
  LOG(info) << __func__;
  print();
#endif
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

int WaveformCalibData::getEntries(int is) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "WaveformCalibData::getEntries channel index %d is out of range", is);
    return 0;
  }
  return 0; // TODO
}

void WaveformCalibData::setN(int n)
{
  if (n >= 0 && n < NBT) {
    mN = n;
    for (int is = 0; is < NChannels; is++) {
      mFirstValid[is] = 0;
      mLastValid[is] = NBT * NTimeBinsPerBC * TSN - 1;
    }
  } else {
    LOG(fatal) << "WaveformCalibData " << __func__ << " wrong stored b.c. setting " << n << " not in range [0:" << WaveformCalibConfig::NBT << "]";
  }
}

int WaveformCalibData::write(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    if (mEntries[is] > 0) {
      TString n = TString::Format("h%d", is);
      TString t = TString::Format("Waveform %d %s", is, ChannelNames[is].data());
      int nbx = mLastValid[is] - mFirstValid[is] + 1;
      int min = mFirstValid[is] - mPeak - 0.5;
      int max = mLastValid[is] - mPeak + 0.5;
      TH1F h(n, t, nbx, min, max);
      for (int ibx = 0; ibx < nbx; ibx++) {
        h.SetBinContent(ibx + 1, mWave[is][mFirstValid[is] + ibx]);
      }
      h.SetEntries(mEntries[is]);
      h.Write("", TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}

void WaveformCalibData::clear()
{
  mCTimeBeg = 0;
  mCTimeEnd = 0;
  mN = 0;
  mPeak = 0;
  for (int32_t is = 0; is < NChannels; is++) {
    mEntries[is] = 0;
    mFirstValid[is] = -1;
    mLastValid[is] = -1;
    for (int iw = 0; iw < NW; iw++) {
      mWave[is][iw] = 0;
    }
  }
}

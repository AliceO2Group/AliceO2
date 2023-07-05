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

#include <TMath.h>
#include <TH1.h>
#include <TFile.h>
#include <TDirectory.h>
#include "Framework/Logger.h"
#include "ZDCCalib/NoiseCalibData.h"

using namespace o2::zdc;

//______________________________________________________________________________
void NoiseCalibData::print() const
{
  LOGF(info, "NoiseCalibData::print ts (%llu:%llu)", mCTimeBeg, mCTimeEnd);
  for (int is = 0; is < NChannels; is++) {
    uint64_t en = 0;
    double mean = 0, var = 0;
    if (getStat(is, en, mean, var) == 0) {
      LOGF(info, "Noise %2d %s with %10llu events %smean %8.1f rms %8.1f (ch) max=%d", is, ChannelNames[is].data(), en, mHisto[is].mOverflow ? " IN_OVERFLOW" : "",
           mean, TMath::Sqrt(var), mHisto[is].getMaxBin());
    }
  }
}

//______________________________________________________________________________
NoiseCalibData& NoiseCalibData::operator+=(const NoiseCalibData& other)
{
  // Adding data in the working representation
  if (other.mOverflow) {
    // Refusing to add an overflow
    LOG(warn) << "NoiseCalibData" << __func__ << " Refusing to add an overflow. BREAK!";
    return *this;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    if (other.mHisto[is].mOverflow) {
      // Refusing to add an overflow histogram
      LOG(warn) << "NoiseCalibData::" << __func__ << " Refusing to add an overflow histogram for ch " << is << " BREAK!";
      return *this;
    }
    // Check if sum will result into an overflow
    // The number of bins is not defined a priori: identify max bin number
    uint32_t maxc = mHisto[is].getMaxBin();
    uint32_t maxo = other.mHisto[is].getMaxBin();
    auto max = std::min(maxc, maxo) + 1;
    for (int32_t i = 0; i < max; i++) {
      auto pcur = mHisto[is].mData.find(i);
      auto poth = other.mHisto[is].mData.find(i);
      if (pcur != mHisto[is].mData.end() && poth != other.mHisto[is].mData.end()) {
        uint64_t sum = pcur->first + pcur->second;
        if (sum > 0xffffffff) {
          LOG(warn) << "NoiseCalibData::" << __func__ << " Addition would result in an overflow for ch " << is << " BREAK!";
          return *this;
        }
      }
    }
  }
  // No problems with overflow
  for (int32_t is = 0; is < NChannels; is++) {
    for (const auto& [key, value] : other.mHisto[is].mData) {
      mHisto[is].mData[key] = mHisto[is].mData[key] + value;
    }
  }
  if (mCTimeBeg == 0 || other.mCTimeBeg < mCTimeBeg) {
    mCTimeBeg = other.mCTimeBeg;
  }
  if (other.mCTimeEnd > mCTimeEnd) {
    mCTimeEnd = other.mCTimeEnd;
  }
#ifdef O2_ZDC_DEBUG
  LOG(info) << __func__;
  print();
#endif
  return *this;
}

//______________________________________________________________________________
NoiseCalibData& NoiseCalibData::operator=(const NoiseCalibSummaryData& s)
{
  mCTimeBeg = s.mCTimeBeg;
  mCTimeEnd = s.mCTimeEnd;
  mOverflow = s.mOverflow;
  for (int32_t is = 0; is < NChannels; is++) {
    // Need to clear all bins since summary data have info only on filled channels
    mHisto[is].clear();
    mHisto[is].mOverflow = s.mOverflowCh[is];
    // Cross check
    if (mHisto[is].mOverflow && mOverflow == false) {
      LOG(warn) << "Overflow bit set on signal " << is;
      mOverflow = true;
    }
  }
  for (const NoiseCalibBinData& bin : s.mData) {
    mHisto[bin.id()].mData[bin.bin()] = bin.cont;
  }
  return *this;
}

//______________________________________________________________________________
NoiseCalibData& NoiseCalibData::operator+=(const NoiseCalibSummaryData* s)
{
  if (s == nullptr) {
    LOG(error) << "NoiseCalibData::operator+=(const NoiseCalibSummaryData* s): null pointer";
    return *this;
  }
  if (s->mOverflow) {
    // Refusing to add an overflow
    LOG(warn) << __func__ << " Refusing to add an overflow NoiseCalibSummaryData BREAK!";
    s->print();
    return *this;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    if (s->mOverflowCh[is]) {
      // Refusing to add an overflow histogram
      LOG(warn) << __func__ << " Refusing to add an overflow histogram for ch " << is << " BREAK!";
      s->print();
      return *this;
    }
  }
  // Check if sum will result into an overflow
  for (auto& bin : s->mData) {
    uint64_t sum = mHisto[bin.id()].mData[bin.bin()] + bin.cont;
    if (sum > 0xffffffff) {
      LOG(warn) << __func__ << " Addition would result in an overflow for ch " << bin.id() << " BREAK!";
      return *this;
    }
  }
  if (mCTimeBeg == 0 || s->mCTimeBeg < mCTimeBeg) {
    mCTimeBeg = s->mCTimeBeg;
  }
  if (s->mCTimeEnd > mCTimeEnd) {
    mCTimeEnd = s->mCTimeEnd;
  }
  for (auto& bin : s->mData) {
    mHisto[bin.id()].mData[bin.bin()] = mHisto[bin.id()].mData[bin.bin()] + bin.cont;
  }
  return *this;
}

//______________________________________________________________________________
void NoiseCalibData::setCreationTime(uint64_t ctime)
{
  mCTimeBeg = ctime;
  mCTimeEnd = ctime;
#ifdef O2_ZDC_DEBUG
  LOGF(info, "NoiseCalibData::setCreationTime %llu", ctime);
#endif
}

//______________________________________________________________________________
uint64_t NoiseCalibData::getEntries(int is) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "NoiseCalibData::getEntries channel index %d is out of range", is);
    return 0;
  }
  return mHisto[is].getEntries();
}

uint64_t NoiseCalibChData::getEntries() const
{
  uint64_t sum = 0;
  for (const auto& [key, value] : mData) {
    sum = sum + value;
  }
  return sum;
}

uint32_t NoiseCalibData::getMaxBin(int is) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "NoiseCalibData::getMaxBin channel index %d is out of range", is);
    return 0;
  }
  return mHisto[is].getMaxBin();
}

uint32_t NoiseCalibChData::getMaxBin() const
{
  uint32_t max = 0;
  for (const auto& [key, value] : mData) {
    if (key > max) {
      max = key;
    }
  }
  return max;
}

//______________________________________________________________________________
int NoiseCalibData::getStat(int is, uint64_t& en, double& mean, double& var) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "NoiseCalibData::getStat channel index %d is out of range", is);
    return 1;
  }
  return mHisto[is].getStat(en, mean, var);
}

int NoiseCalibChData::getStat(uint64_t& en, double& mean, double& var) const
{
  en = 0;
  uint64_t sum = 0;
  for (const auto& [key, value] : mData) {
    en = en + value;
    sum = sum + key * value;
  }
  if (en == 0) {
    return 1;
  }
  mean = double(sum) / double(en);
  if (en > 1) {
    double sums = 0;
    for (const auto& [key, value] : mData) {
      double diff = key - mean;
      sums = sums + (value * diff * diff);
    }
    var = sums / (en - 1);
  } else {
    var = 0;
  }
  // Convert mean to correct range
  mean = mean;
  return 0;
}

//______________________________________________________________________________
int NoiseCalibData::saveDebugHistos(const std::string fn)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  double factor = 1. / double(NTimeBinsPerBC * (NTimeBinsPerBC - 1));
  for (int32_t is = 0; is < NChannels; is++) {
    uint64_t nen = mHisto[is].getEntries();
    if (nen > 0) {
      int32_t max = mHisto[is].getMaxBin();
      TString n = TString::Format("h%d", is);
      TString t = TString::Format("Noise %d %s", is, ChannelNames[is].data());
      TH1F h(n, t, max + 1, -0.5 * factor, (max + 0.5) * factor);
      for (int ibx = 0; ibx < max; ibx++) {
        h.SetBinContent(ibx + 1, mHisto[is].mData[ibx] * factor);
      }
      h.SetEntries(nen);
      h.Write("", TObject::kOverwrite);
    }
  }
  f->Close();
  cwd->cd();
  return 0;
}

//______________________________________________________________________________
void NoiseCalibData::clear()
{
  mCTimeBeg = 0;
  mCTimeEnd = 0;
  mOverflow = false;
  mSummary.clear();
  for (int32_t is = 0; is < NChannels; is++) {
    mHisto[is].clear();
  }
}

void NoiseCalibChData::clear()
{
  mOverflow = false;
  mData.clear();
}

void NoiseCalibSummaryData::clear()
{
  mCTimeBeg = 0;
  mCTimeEnd = 0;
  mOverflow = false;
  mOverflowCh.fill(false);
  mData.clear();
}

//______________________________________________________________________________
NoiseCalibSummaryData& NoiseCalibData::getSummary()
{
  mSummary.clear();
  mSummary.mCTimeBeg = mCTimeBeg;
  mSummary.mCTimeEnd = mCTimeEnd;
  mSummary.mOverflow = mOverflow;
  for (int32_t is = 0; is < NChannels; is++) {
    mSummary.mOverflowCh[is] = mHisto[is].mOverflow;
    for (const auto& [ib, cont] : mHisto[is].mData) {
      if (cont > 0) {
        mSummary.mData.emplace_back(is, ib, cont);
      }
    }
  }
  return mSummary;
}

//______________________________________________________________________________
void NoiseCalibSummaryData::print() const
{
  LOGF(info, "NoiseCalibSummaryData: %llu:%llu %d bins%s", mCTimeBeg, mCTimeEnd, mData.size(), (mOverflow ? " OVERFLOW_BIT" : ""));
  if (mOverflow) {
    printf("OVERFLOW:");
    for (int ich = 0; ich < NChannels; ich++) {
      if (mOverflowCh[ich]) {
        printf(" %s", ChannelNames[ich].data());
      }
    }
    printf("\n");
  }
  int nbin[NChannels] = {0};
  uint64_t ccont[NChannels] = {0};
  for (auto& bin : mData) {
    nbin[bin.id()] = nbin[bin.id()] + 1;
    ccont[bin.id()] = ccont[bin.id()] + bin.cont;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    LOG(info) << "Summary ch " << is << " nbin = " << nbin[is] << " ccont = " << ccont[is];
  }
}

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
#include "ZDCCalib/BaselineCalibData.h"

using namespace o2::zdc;

//______________________________________________________________________________
void BaselineCalibData::print() const
{
  LOGF(info, "BaselineCalibData::print ts (%llu:%llu)", mCTimeBeg, mCTimeEnd);
  for (int is = 0; is < NChannels; is++) {
    uint64_t en = 0;
    double mean = 0, var = 0;
    if (getStat(is, en, mean, var) == 0) {
      LOGF(info, "Baseline %2d %s with %10llu events mean %8.1f rms %8.1f (ch)", is, ChannelNames[is].data(), en, mean, TMath::Sqrt(var));
    }
  }
}

//______________________________________________________________________________
BaselineCalibData& BaselineCalibData::operator+=(const BaselineCalibData& other)
{
  if (other.mOverflow) {
    // Refusing to add an overflow
    LOG(warn) << "BaselineCalibData" << __func__ << " Refusing to add an overflow. BREAK!";
    return *this;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    if (other.mHisto[is].mOverflow) {
      // Refusing to add an overflow histogram
      LOG(warn) << "BaselineCalibData::" << __func__ << " Refusing to add an overflow histogram for ch " << is << " BREAK!";
      return *this;
    }
    // Check if sum will result into an overflow
    for (int32_t i = 0; i < BaselineCalibChData::NW; i++) {
      uint64_t sum = mHisto[is].mData[i] + other.mHisto[is].mData[i];
      if (sum > 0xffffffff) {
        LOG(warn) << "BaselineCalibData::" << __func__ << " Addition would result in an overflow for ch " << is << " BREAK!";
        return *this;
      }
    }
  }
  // No problems with overflow
  for (int32_t is = 0; is < NChannels; is++) {
    for (int32_t i = 0; i < BaselineCalibChData::NW; i++) {
      mHisto[is].mData[i] = mHisto[is].mData[i] + other.mHisto[is].mData[i];
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
BaselineCalibData& BaselineCalibData::operator=(const BaselineCalibSummaryData& s)
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
  for (const BaselineCalibBinData& bin : s.mData) {
    mHisto[bin.id].mData[bin.ibin] = bin.cont;
  }
  return *this;
}

//______________________________________________________________________________
BaselineCalibData& BaselineCalibData::operator+=(const BaselineCalibSummaryData* s)
{
  if (s == nullptr) {
    LOG(error) << "BaselineCalibData::operator+=(const BaselineCalibSummaryData* s): null pointer";
    return *this;
  }
  if (s->mOverflow) {
    // Refusing to add an overflow
    LOG(warn) << __func__ << " Refusing to add an overflow BaselineCalibSummaryData BREAK!";
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
    uint64_t sum = mHisto[bin.id].mData[bin.ibin] = bin.cont;
    if (sum > 0xffffffff) {
      LOG(warn) << __func__ << " Addition would result in an overflow for ch " << bin.id << " BREAK!";
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
    mHisto[bin.id].mData[bin.ibin] += bin.cont;
  }
  return *this;
}

//______________________________________________________________________________
void BaselineCalibData::setCreationTime(uint64_t ctime)
{
  mCTimeBeg = ctime;
  mCTimeEnd = ctime;
#ifdef O2_ZDC_DEBUG
  LOGF(info, "BaselineCalibData::setCreationTime %llu", ctime);
#endif
}

//______________________________________________________________________________
uint64_t BaselineCalibData::getEntries(int is) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "BaselineCalibData::getEntries channel index %d is out of range", is);
    return 0;
  }
  return mHisto[is].getEntries();
}

uint64_t BaselineCalibChData::getEntries() const
{
  uint64_t sum = 0;
  for (int32_t i = 0; i < NW; i++) {
    sum += mData[i];
  }
  return sum;
}

//______________________________________________________________________________
int BaselineCalibData::getStat(int is, uint64_t& en, double& mean, double& var) const
{
  if (is < 0 || is >= NChannels) {
    LOGF(error, "BaselineCalibData::getStat channel index %d is out of range", is);
    return 1;
  }
  return mHisto[is].getStat(en, mean, var);
}

int BaselineCalibChData::getStat(uint64_t& en, double& mean, double& var) const
{
  en = 0;
  uint64_t sum = 0;
  for (uint64_t i = 0; i < NW; i++) {
    en += mData[i];
    sum += i * mData[i];
  }
  if (en == 0) {
    return 1;
  }
  mean = double(sum) / double(en);
  if (en > 1) {
    double sums = 0;
    for (int32_t i = 0; i < NW; i++) {
      double diff = i - mean;
      sums += (mData[i] * diff * diff);
    }
    var = sums / (en - 1);
  } else {
    var = 0;
  }
  // Convert mean to correct range
  mean = mean + BaselineMin;
  return 0;
}

//______________________________________________________________________________
int BaselineCalibData::saveDebugHistos(const std::string fn, float factor = 1)
{
  TDirectory* cwd = gDirectory;
  TFile* f = new TFile(fn.data(), "recreate");
  if (f->IsZombie()) {
    LOG(error) << "Cannot create file: " << fn;
    return 1;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    uint64_t nen = mHisto[is].getEntries();
    if (nen > 0) {
      TString n = TString::Format("h%d", is);
      TString t = TString::Format("Baseline %d %s", is, ChannelNames[is].data());
      TH1F h(n, t, BaselineRange, (BaselineMin - 0.5) * factor, (BaselineMax + 0.5) * factor);
      for (int ibx = 0; ibx < BaselineRange; ibx++) {
        h.SetBinContent(ibx + 1, mHisto[is].mData[ibx]);
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
void BaselineCalibData::clear()
{
  mCTimeBeg = 0;
  mCTimeEnd = 0;
  mOverflow = false;
  mSummary.clear();
  for (int32_t is = 0; is < NChannels; is++) {
    mHisto[is].clear();
  }
}

void BaselineCalibChData::clear()
{
  mOverflow = false;
  for (int iw = 0; iw < NW; iw++) {
    mData[iw] = 0;
  }
}

void BaselineCalibSummaryData::clear()
{
  mCTimeBeg = 0;
  mCTimeEnd = 0;
  mOverflow = false;
  mOverflowCh.fill(false);
  mData.clear();
}

//______________________________________________________________________________
BaselineCalibSummaryData& BaselineCalibData::getSummary()
{
  mSummary.clear();
  mSummary.mCTimeBeg = mCTimeBeg;
  mSummary.mCTimeEnd = mCTimeEnd;
  mSummary.mOverflow = mOverflow;
  for (int32_t is = 0; is < NChannels; is++) {
    mSummary.mOverflowCh[is] = mHisto[is].mOverflow;
    for (int32_t ib = 0; ib < BaselineRange; ib++) {
      uint32_t cont = mHisto[is].mData[ib];
      if (cont > 0) {
        mSummary.mData.emplace_back(is, ib, cont);
      }
    }
  }
  return mSummary;
}

//______________________________________________________________________________
void BaselineCalibSummaryData::print() const
{
  LOGF(info, "BaselineCalibSummaryData: %llu:%llu %d bins%s", mCTimeBeg, mCTimeEnd, mData.size(), (mOverflow ? " OVERFLOW_BIT" : ""));
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
    nbin[bin.id]++;
    ccont[bin.id] += bin.cont;
  }
  for (int32_t is = 0; is < NChannels; is++) {
    LOG(info) << "Summary ch " << is << " nbin = " << nbin[is] << " ccont = " << ccont[is];
  }
}

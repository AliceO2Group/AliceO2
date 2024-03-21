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

#include <iostream>
#include "EMCALCalibration/PedestalProcessorData.h"

using namespace o2::emcal;

PedestalProcessorData& PedestalProcessorData::operator+=(const PedestalProcessorData& other)
{
  for (std::size_t ichan{0}; ichan < mDataFECHG.size(); ++ichan) {
    mDataFECHG[ichan] += other.mDataFECHG[ichan];
    mDataFECLG[ichan] += other.mDataFECLG[ichan];
  }
  for (std::size_t ichan{0}; ichan < mDataLEDMONHG.size(); ++ichan) {
    mDataLEDMONHG[ichan] += other.mDataLEDMONHG[ichan];
    mDataLEDMONLG[ichan] += other.mDataLEDMONLG[ichan];
  }
  return *this;
}

void PedestalProcessorData::fillADC(unsigned short adc, unsigned short tower, bool lowGain, bool LEDMON)
{
  auto maxentries = LEDMON ? mDataLEDMONHG.size() : mDataFECHG.size();
  if (tower >= maxentries) {
    throw ChannelIndexException(tower, maxentries);
  }
  if (LEDMON) {
    if (lowGain) {
      mDataLEDMONLG[tower].add(adc);
    } else {
      mDataLEDMONHG[tower].add(adc);
    }
  } else {
    if (lowGain) {
      mDataFECLG[tower].add(adc);
    } else {
      mDataFECHG[tower].add(adc);
    }
  }
}

PedestalProcessorData::PedestalValue PedestalProcessorData::getValue(unsigned short tower, bool lowGain, bool LEDMON) const
{
  auto maxentries = LEDMON ? mDataLEDMONHG.size() : mDataFECHG.size();
  if (tower >= maxentries) {
    throw ChannelIndexException(tower, maxentries);
  }
  float mean, rms;
  if (LEDMON) {
    if (lowGain) {
      return mDataLEDMONLG[tower].getMeanRMS2<double>();
    } else {
      return mDataLEDMONHG[tower].getMeanRMS2<double>();
    }
  } else {
    if (lowGain) {
      return mDataFECLG[tower].getMeanRMS2<double>();
    } else {
      return mDataFECHG[tower].getMeanRMS2<double>();
    }
  }
}

int PedestalProcessorData::getEntriesForChannel(unsigned short tower, bool lowGain, bool LEDMON) const
{
  auto maxentries = LEDMON ? mDataLEDMONHG.size() : mDataFECHG.size();
  if (tower >= maxentries) {
    throw ChannelIndexException(tower, maxentries);
  }
  float mean, rms;
  if (LEDMON) {
    if (lowGain) {
      return mDataLEDMONLG[tower].n;
    } else {
      return mDataLEDMONHG[tower].n;
    }
  } else {
    if (lowGain) {
      return mDataFECLG[tower].n;
    } else {
      return mDataFECHG[tower].n;
    }
  }
}

void PedestalProcessorData::reset()
{
  for (auto& acc : mDataFECHG) {
    acc.clear();
  }
  for (auto& acc : mDataFECLG) {
    acc.clear();
  }
  for (auto& acc : mDataLEDMONHG) {
    acc.clear();
  }
  for (auto& acc : mDataLEDMONLG) {
    acc.clear();
  }
}

PedestalProcessorData o2::emcal::operator+(const PedestalProcessorData& lhs, const PedestalProcessorData& rhs)
{
  PedestalProcessorData result(lhs);
  result += rhs;
  return result;
}

std::ostream& o2::emcal::operator<<(std::ostream& stream, const PedestalProcessorData::ChannelIndexException& ex)
{
  stream << ex.what();
  return stream;
}
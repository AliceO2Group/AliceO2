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
#include <algorithm>
#include "FOCALReconstruction/PadData.h"

using namespace o2::focal;

ASICData::ASICData(ASICHeader firstheader, ASICHeader secondheader)
{
  setFirstHeader(firstheader);
  setSecondHeader(secondheader);
}

void ASICData::setHeader(ASICHeader header, int index)
{
  if (index >= NHALVES) {
    throw IndexException(index, NHALVES);
  }
  mHeaders[index] = header;
}

void ASICData::setChannel(ASICChannel data, int index)
{
  if (index >= NCHANNELS) {
    throw IndexException(index, NCHANNELS);
  }
  mChannels[index] = data;
}

void ASICData::setChannels(const gsl::span<const ASICChannel> channels)
{
  std::copy(channels.begin(), channels.end(), mChannels.begin());
}

void ASICData::setCMN(ASICChannel data, int index)
{
  if (index >= NHALVES) {
    throw IndexException(index, NHALVES);
  }
  mCMNChannels[index] = data;
}

void ASICData::setCMNs(const gsl::span<const ASICChannel> channels)
{
  std::copy(channels.begin(), channels.end(), mCMNChannels.begin());
}

void ASICData::setCalib(ASICChannel data, int index)
{
  if (index >= NHALVES) {
    throw IndexException(index, NHALVES);
  }
  mCalibChannels[index] = data;
}

void ASICData::setCalibs(const gsl::span<const ASICChannel> channels)
{
  std::copy(channels.begin(), channels.end(), mCalibChannels.begin());
}

ASICHeader ASICData::getHeader(int index) const
{
  if (index >= NHALVES) {
    throw IndexException(index, NHALVES);
  }
  return mHeaders[index];
}

gsl::span<const ASICHeader> ASICData::getHeaders() const
{
  return mHeaders;
}

gsl::span<const ASICChannel> ASICData::getChannels() const
{
  return mChannels;
}

ASICChannel ASICData::getChannel(int index) const
{
  if (index >= NCHANNELS) {
    throw IndexException(index, NCHANNELS);
  }
  return mChannels[index];
}

ASICChannel ASICData::getCalib(int index) const
{
  if (index >= NHALVES) {
    throw IndexException(index, NHALVES);
  }
  return mCalibChannels[index];
}

gsl::span<const ASICChannel> ASICData::getCalibs() const
{
  return mCalibChannels;
}

ASICChannel ASICData::getCMN(int index) const
{
  if (index >= NCHANNELS) {
    throw IndexException(index, NCHANNELS);
  }
  return mCMNChannels[index];
}

gsl::span<const ASICChannel> ASICData::getCMNs() const
{
  return mCMNChannels;
}

void ASICData::reset()
{
  /*
  for (auto& header : mHeaders) {
    header.mData = 0;
  }
  for (auto& channel : mChannels) {
    channel.mData = 0;
  }
  for (auto& channel : mCalibChannels) {
    channel.mData = 0;
  }
  for (auto& channel : mCMNChannels) {
    channel.mData = 0;
  }
  */
  std::fill(mHeaders.begin(), mHeaders.end(), ASICHeader(0));
  std::fill(mChannels.begin(), mChannels.end(), ASICChannel(0));
  std::fill(mCalibChannels.begin(), mCalibChannels.end(), ASICChannel(0));
  std::fill(mCMNChannels.begin(), mCMNChannels.end(), ASICChannel(0));
}

gsl::span<const TriggerWord> ASICContainer::getTriggerWords() const
{
  return mTriggerData;
}

void ASICContainer::appendTriggerWords(gsl::span<const TriggerWord> triggerwords)
{
  std::copy(triggerwords.begin(), triggerwords.end(), std::back_inserter(mTriggerData));
}

void ASICContainer::appendTriggerWord(TriggerWord triggerword)
{
  mTriggerData.emplace_back(triggerword);
}

void ASICContainer::reset()
{
  mASIC.reset();
  mTriggerData.clear();
}

void PadData::reset()
{
  for (auto& asic : mASICs) {
    asic.reset();
  }
}

const ASICContainer& PadData::getDataForASIC(int index) const
{
  if (index >= NASICS) {
    throw IndexException(index, NASICS);
  }
  return mASICs[index];
}

ASICContainer& PadData::getDataForASIC(int index)
{
  if (index >= NASICS) {
    throw IndexException(index, NASICS);
  }
  return mASICs[index];
}
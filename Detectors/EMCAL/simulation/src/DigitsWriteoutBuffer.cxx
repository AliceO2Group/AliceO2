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

#include <unordered_map>
#include <vector>
#include <list>
#include <deque>
#include <iostream>
#include <gsl/span>
#include "EMCALSimulation/LabeledDigit.h"
#include "EMCALSimulation/DigitsWriteoutBuffer.h"

using namespace o2::emcal;

DigitsWriteoutBuffer::DigitsWriteoutBuffer(unsigned int nTimeBins, unsigned int binWidth) : mBufferSize(nTimeBins),
                                                                                            mTimeBinWidth(binWidth)
{
  for (int itime = 0; itime < nTimeBins; itime++) {
    mTimedDigits.push_back(std::unordered_map<int, std::list<LabeledDigit>>());
  }
}

void DigitsWriteoutBuffer::clear()
{
  std::cout << "Clearing ..." << std::endl;
  mTimedDigits.clear();
}

void DigitsWriteoutBuffer::reserve()
{
  //Fill in the missing entries
  for (int itime = 0; itime < (mBufferSize - mTimedDigits.size()); itime++) {
    mTimedDigits.push_back(std::unordered_map<int, std::list<LabeledDigit>>());
  }
}

void DigitsWriteoutBuffer::addDigit(unsigned int towerID, LabeledDigit dig, double eventTime)
{
  int nsamples = int(eventTime / mTimeBinWidth);

  if (nsamples >= mTimedDigits.size()) {
    mTimedDigits.pop_front();
    mTimedDigits.push_back(std::unordered_map<int, std::list<LabeledDigit>>());
    nsamples = mTimedDigits.size() - 1;
  }

  auto& timeEntry = mTimedDigits[nsamples];

  auto towerEntry = timeEntry.find(towerID);
  if (towerEntry == timeEntry.end()) {
    towerEntry = timeEntry.insert(std::pair<int, std::list<o2::emcal::LabeledDigit>>(towerID, std::list<o2::emcal::LabeledDigit>())).first;
  }

  towerEntry->second.push_back(dig);

  mLastEventTime = eventTime;
}

std::list<std::unordered_map<int, std::list<LabeledDigit>>> DigitsWriteoutBuffer::getLastNSamples(int nsamples)
{
  int startingPosition = int(mLastEventTime / mTimeBinWidth) - nsamples;
  if (startingPosition < 0) {
    startingPosition = 0;
  }

  std::list<std::unordered_map<int, std::list<LabeledDigit>>> output;

  for (int ientry = startingPosition; ientry < startingPosition + nsamples; ientry++) {
    output.push_back(mTimedDigits[ientry]);
  }
  return output;
  //return gsl::span<std::unordered_map<int, std::list<LabeledDigit>>>(&mTimedDigits[startingPosition], nsamples);
}

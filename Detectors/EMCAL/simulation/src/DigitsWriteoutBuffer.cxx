// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <unordered_map>
#include <vector>
#include <list>
#include <deque>
#include <gsl/span>
#include "EMCALSimulation/LabeledDigit.h"
#include "EMCALSimulation/DigitsWriteoutBuffer.h"

using namespace o2::emcal;

DigitsWriteoutBuffer::DigitsWriteoutBuffer(unsigned int nTimeBins, unsigned int binWidth) : mBufferSize(nTimeBins),
                                                                                            mTimeBinWidth(binWidth)
{
  mTimedDigits.resize(nTimeBins);
  mMarker.mReferenceTime = 0.;
  mMarker.mPositionInBuffer = mTimedDigits.begin();
}

void DigitsWriteoutBuffer::clear()
{
  mTimedDigits.clear();
  mMarker.mReferenceTime = 0.;
  mMarker.mPositionInBuffer = mTimedDigits.begin();
}

void DigitsWriteoutBuffer::addDigit(unsigned int towerID, LabeledDigit dig, double eventTime)
{

  int nsamples = int((eventTime - mMarker.mReferenceTime) / mTimeBinWidth);
  auto timeEntry = mMarker.mPositionInBuffer;
  timeEntry[nsamples][towerID].push_back(dig);
}

void DigitsWriteoutBuffer::forwardMarker(double eventTime)
{
  mMarker.mReferenceTime = eventTime;
  mMarker.mPositionInBuffer++;

  // Allocate new memory at the end
  mTimedDigits.push_back(std::unordered_map<int, std::list<LabeledDigit>>());

  // Drop entry at the front, because it is outside the current readout window
  // only done if we have at least 15 entries
  if (mMarker.mPositionInBuffer - mTimedDigits.begin() > mNumberReadoutSamples) {
    mTimedDigits.pop_front();
  }
}

gsl::span<std::unordered_map<int, std::list<LabeledDigit>>> DigitsWriteoutBuffer::getLastNSamples(int nsamples)
{
  return gsl::span<std::unordered_map<int, std::list<LabeledDigit>>>(&mTimedDigits[int(mMarker.mPositionInBuffer - mTimedDigits.begin() - nsamples)], nsamples);
}

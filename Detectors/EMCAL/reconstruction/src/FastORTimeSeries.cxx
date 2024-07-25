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
#include <iostream>
#include "EMCALReconstruction/FastORTimeSeries.h"
#include "EMCALBase/TRUDecodingErrors.h"

using namespace o2::emcal;

void FastORTimeSeries::fillReversed(const gsl::span<const uint16_t> samples, uint8_t starttime)
{
  if (starttime >= 14) {
    throw FastOrStartTimeInvalidException(starttime);
  }
  for (std::size_t isample = 0; isample < samples.size(); isample++) {
    mTimeSamples[starttime - isample] = samples[isample];
  }
}

uint16_t FastORTimeSeries::calculateL1TimeSum(uint8_t l0time) const
{
  uint16_t timesum = 0;
  int firstbin = l0time - 4; // Include sample before the L0 time
  for (int isample = firstbin; isample < firstbin + 4; isample++) {
    timesum += mTimeSamples[isample];
  }
  return timesum;
}

void FastORTimeSeries::setSize(int maxsamples)
{
  mTimeSamples.resize(maxsamples);
}

void FastORTimeSeries::clear()
{
  std::fill(mTimeSamples.begin(), mTimeSamples.end(), 0);
}
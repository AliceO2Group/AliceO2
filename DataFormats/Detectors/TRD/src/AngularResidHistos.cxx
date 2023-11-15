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

/// \file AngularResidHistos.cxx
/// \brief Class to store the output of the global tracking based TRD calibration

#include "DataFormatsTRD/AngularResidHistos.h"
#include <fairlogger/Logger.h>
#include <cmath>

using namespace o2::trd;
using namespace o2::trd::constants;

void AngularResidHistos::reset()
{
  mHistogramEntries.fill(0);
  mNEntriesPerBin.fill(0);
  mNEntriesTotal = 0;
}

bool AngularResidHistos::addEntry(float deltaAlpha, float impactAngle, int chamberId)
{
  // add entry for given angular residual
  // returns 0 in case of success (impact angle is in valid range)
  int chamberOffset = chamberId * NBINSANGLEDIFF;
  if (std::fabs(impactAngle) < MAXIMPACTANGLE) {
    int iBin = (impactAngle + MAXIMPACTANGLE) * INVBINWIDTH;
    mHistogramEntries[chamberOffset + iBin] += deltaAlpha;
    ++mNEntriesPerBin[chamberOffset + iBin];
    ++mNEntriesTotal;
  } else {
    LOG(debug) << "Under-/overflow entry detected for impact angle " << impactAngle;
    return 1;
  }
  return 0;
}

void AngularResidHistos::fill(const AngularResidHistos& input)
{
  for (int i = 0; i < MAXCHAMBER * NBINSANGLEDIFF; ++i) {
    mHistogramEntries[i] += input.getHistogramEntry(i);
    mNEntriesPerBin[i] += input.getBinCount(i);
    mNEntriesTotal += input.getBinCount(i);
  }
}

void AngularResidHistos::merge(const AngularResidHistos* prev)
{
  for (int i = 0; i < MAXCHAMBER * NBINSANGLEDIFF; ++i) {
    mHistogramEntries[i] += prev->getHistogramEntry(i);
    mNEntriesPerBin[i] += prev->getBinCount(i);
    mNEntriesTotal += prev->getBinCount(i);
  }
}

void AngularResidHistos::print()
{
  LOG(info) << "There are " << mNEntriesTotal << " entries in the container";
  for (int i = 0; i < MAXCHAMBER * NBINSANGLEDIFF; ++i) {
    if (mNEntriesPerBin[i] != 0) {
      LOGF(info, "Global bin %i has %i entries. Average angular residual: %f", i, mNEntriesPerBin[i], mHistogramEntries[i] / mNEntriesPerBin[i]);
    }
  }
}

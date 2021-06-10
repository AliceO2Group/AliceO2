// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

bool AngularResidHistos::addEntry(float deltaAlpha, float impactAngle, int chamberId)
{
  // add entry for given angular residual
  // returns 0 in case of success (impact angle is in valid range)
  int chamberOffset = chamberId * NBINSANGLEDIFF;
  if (std::fabs(impactAngle) >= MAXIMPACTANGLE) {
    LOG(DEBUG) << "Under-/overflow entry detected for impact angle " << impactAngle;
    return 1;
  } else {
    int iBin = (impactAngle + MAXIMPACTANGLE) * INVBINWIDTH;
    mHistogramEntries[chamberOffset + iBin] += deltaAlpha;
    ++mNEntriesPerBin[chamberOffset + iBin];
    ++mNEntriesTotal;
  }
  return 0;
}

void AngularResidHistos::fill(const gsl::span<const AngularResidHistos> input)
{
  for (const auto& data : input) {
    for (int i = 0; i < MAXCHAMBER * NBINSANGLEDIFF; ++i) {
      mHistogramEntries[i] += data.getHistogramEntry(i);
      mNEntriesPerBin[i] += data.getBinCount(i);
      mNEntriesTotal += data.getBinCount(i);
    }
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
  LOG(INFO) << "There are " << mNEntriesTotal << " entries in the container";
  for (int i = 0; i < MAXCHAMBER * NBINSANGLEDIFF; ++i) {
    if (mNEntriesPerBin[i] != 0) {
      LOGF(INFO, "Global bin %i has %i entries. Average angular residual: %f", i, mNEntriesPerBin[i], mHistogramEntries[i]);
    }
  }
}

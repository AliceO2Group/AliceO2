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

//ClassImp(o2::trd::AngularResidHistos);

void AngularResidHistos::addEntry(float deltaAlpha, float impactAngle, int chamberId)
{
  int chamberOffset = chamberId * NBINSANGLEDIFF;
  if (std::fabs(impactAngle) >= MAXIMPACTANGLE) {
    // make over-/undeflow bin entry
    mHistogramEntries[chamberOffset + NBINSANGLEDIFF] += deltaAlpha;
    ++mNEntriesPerBin[chamberOffset + NBINSANGLEDIFF];
  } else {
    int iBin = (impactAngle + MAXIMPACTANGLE) * INVBINWIDTH;
    mHistogramEntries[chamberOffset + iBin] += deltaAlpha;
    ++mNEntriesPerBin[chamberOffset + iBin];
    ++mNEntriesTotal;
  }
}

void AngularResidHistos::fill(const gsl::span<const AngularResidHistos> input)
{
  for (const auto& data : input) {
    for (int i = 0; i < MAXCHAMBER * (NBINSANGLEDIFF + 1); ++i) {
      mHistogramEntries[i] += data.getHistogramEntry(i);
      mNEntriesPerBin[i] += data.getBinCount(i);
      mNEntriesTotal += data.getBinCount(i);
    }
  }
}

void AngularResidHistos::merge(const AngularResidHistos* prev)
{
  for (int i = 0; i < MAXCHAMBER * (NBINSANGLEDIFF + 1); ++i) {
    mHistogramEntries[i] += prev->getHistogramEntry(i);
    mNEntriesPerBin[i] += prev->getBinCount(i);
    mNEntriesTotal += prev->getBinCount(i);
  }
}

void AngularResidHistos::print()
{
  LOG(INFO) << "There are " << mNEntriesTotal << " entries in the container (excluding under-/overflow bin)";
  for (int i = 0; i < MAXCHAMBER * (NBINSANGLEDIFF + 1); ++i) {
    if (mNEntriesPerBin[i] != 0) {
      LOGF(INFO, "Global bin %i has %i entries. Average angular residual: %f", i, mNEntriesPerBin[i], mHistogramEntries[i]);
    }
  }
}

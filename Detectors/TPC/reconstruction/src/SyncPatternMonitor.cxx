// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SyncPatternMonitor.cxx
/// \author Sebastian Klewin

#include "TPCReconstruction/SyncPatternMonitor.h"
#include <iostream>

using namespace o2::tpc;
constexpr std::array<short, 32> SyncPatternMonitor::SYNC_PATTERN;

SyncPatternMonitor::SyncPatternMonitor()
  : SyncPatternMonitor(-1, -1)
{
}

SyncPatternMonitor::SyncPatternMonitor(int sampa, int lowHigh)
  : mPatternFound(false), mPosition(SYNC_START), mHwWithPattern(-1), mSampa(sampa), mLowHigh(lowHigh), mCheckedWords(0)
{
}

SyncPatternMonitor::SyncPatternMonitor(const SyncPatternMonitor& other) = default;

void SyncPatternMonitor::reset()
{
  mPatternFound = false;
  mPosition = SYNC_START;
  mHwWithPattern = -1;
  mCheckedWords = 0;
  LOG(DEBUG) << "Sync pattern monitoring for SAMPA " << mSampa << " (" << ((mLowHigh == 0) ? "low" : "high") << " bits) "
             << "was resetted";
}

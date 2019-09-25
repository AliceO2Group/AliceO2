// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CalibLHCphaseTOF.cxx
/// \brief Class to store the output of the matching to TOF for calibration

#include <algorithm>
#include <cstdio>
#include "DataFormatsTOF/CalibLHCphaseTOF.h"

using namespace o2::dataformats;

//ClassImp(o2::dataformats::CalibLHCphaseTOF);

float CalibLHCphaseTOF::getLHCphase(int timestamp) const
{
  int n = 0;
  while (n < mLHCphase.size() && mLHCphase[n].first < timestamp)
    n++;
  n--;

  if (n < 0) { // timestamp is before of the first available value
    return 0;
  }
  return mLHCphase[n].second;
}
//______________________________________________

void CalibLHCphaseTOF::addLHCphase(int timestamp, float phaseLHC)
{
  // optimized if timestamp are given in increasing order
  int n = mLHCphase.size();
  mLHCphase.emplace_back(timestamp, phaseLHC);

  if (n && mLHCphase[n].first < mLHCphase[n - 1].first) { // in the wrong order sort!
    std::sort(mLHCphase.begin(), mLHCphase.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });
  }
}
//______________________________________________

CalibLHCphaseTOF& CalibLHCphaseTOF::operator+=(const CalibLHCphaseTOF& other)
{
  if (other.mLHCphase.size() > mLHCphase.size()) {
    mLHCphase.clear();
    for (auto obj = other.mLHCphase.begin(); obj != other.mLHCphase.end(); obj++)
      mLHCphase.push_back(*obj);
  }
  return *this;
}
//______________________________________________

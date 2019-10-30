// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "NoiseSuppression.h"

using namespace gpucf;

RowMap<std::vector<Digit>> NoiseSuppression::run(
  const RowMap<std::vector<Digit>>& digits,
  const RowMap<Map<bool>>& isPeak,
  const Map<float>& chargeMap)
{
  RowMap<std::vector<Digit>> filteredPeaks;

  for (size_t row = 0; row < TPC_NUM_OF_ROWS; row++) {
    filteredPeaks[row] = runImpl(digits[row], isPeak[row], chargeMap);
  }

  return filteredPeaks;
}

std::vector<Digit> NoiseSuppression::runOnAllRows(
  View<Digit> digits,
  const Map<bool>& isPeak,
  const Map<float>& chargeMap)
{
  return runImpl(digits, isPeak, chargeMap);
}

// vim: set ts=4 sw=4 sts=4 expandtab:

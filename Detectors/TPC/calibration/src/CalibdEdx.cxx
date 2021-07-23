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

#include "TPCCalibration/CalibdEdx.h"

#include <algorithm>
#include <array>
#include <cstddef>

//o2 includes
#include "TPCCalibration/CalibdEdxHistos.h"

using namespace o2::tpc;

void CalibdEdx::process(const CalibdEdxHistos& histos)
{
  const auto& processHistos = [](const CalibdEdxHistos::Hist& hist) { return hist.getStatisticsData().mCOG; };

  const auto& totHists = histos.getTotEntries();
  std::transform(totHists.begin(), totHists.end(), mTotEntries.begin(), processHistos);

  const auto& maxHists = histos.getMaxEntries();
  std::transform(maxHists.begin(), maxHists.end(), mMaxEntries.begin(), processHistos);
}

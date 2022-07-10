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

#include <cmath>

#include "TPCBase/PadRegionInfo.h"

namespace o2
{
namespace tpc
{

PadRegionInfo::PadRegionInfo(const unsigned char region,
                             const unsigned char partition,
                             const unsigned char numberOfPadRows,
                             const float padHeight,
                             const float padWidth,
                             const float radiusFirstRow,
                             const unsigned char rowOffset,
                             const float xhelper,
                             const unsigned char globalRowOffset)
  : mRegion{region}, mPartition{partition}, mNumberOfPadRows{numberOfPadRows}, mPadHeight{padHeight}, mPadWidth{padWidth}, mInvPadHeight{1.f / padHeight}, mInvPadWidth{1.f / padWidth}, mRadiusFirstRow{radiusFirstRow}, mRowOffset{rowOffset}, mXhelper{xhelper}, mNumberOfPads{0}, mGlobalRowOffset{globalRowOffset}, mPadsPerRow(numberOfPadRows)
{
  init();
}

void PadRegionInfo::init()
{

  const float ks = mPadHeight / mPadWidth * tan(1.74532925199432948e-01); // tan(10deg)
  // initialize number of pads per row
  for (unsigned char irow = 0; irow < mNumberOfPadRows; ++irow) {
    mPadsPerRow[irow] = 2 * std::floor(ks * (irow + mRowOffset) + mXhelper);
    mNumberOfPads += mPadsPerRow[irow];
  }
}

} // namespace tpc
} // namespace o2

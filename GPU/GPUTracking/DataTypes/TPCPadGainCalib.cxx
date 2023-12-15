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

/// \file TPCPadGainCalib.cxx
/// \author Felix Weiglhofer

#include "TPCPadGainCalib.h"

#include "GPUTPCGeometry.h"
#include "DataFormatsTPC/Constants.h"

using namespace GPUCA_NAMESPACE::gpu;

TPCPadGainCalib::TPCPadGainCalib()
{
  GPUTPCGeometry geo{};
  int offset = 0;
  for (int r = 0; r < GPUCA_ROW_COUNT; r++) {
    mPadOffsetPerRow[r] = offset;
    offset += geo.NPads(r);
  }
}

#ifndef GPUCA_STANDALONE
#include "TPCBase/CalDet.h"

TPCPadGainCalib::TPCPadGainCalib(const o2::tpc::CalDet<float>& gainMap) : TPCPadGainCalib()
{
  setFromMap(gainMap);
}

TPCPadGainCalib::TPCPadGainCalib(const o2::tpc::CalDet<float>& gainMap, const float minValue, const float maxValue, const bool inv) : TPCPadGainCalib()
{
  setMinCorrectionFactor(minValue);
  setMaxCorrectionFactor(maxValue);
  setFromMap(gainMap, inv);
}

void TPCPadGainCalib::setFromMap(const o2::tpc::CalDet<float>& gainMap, const bool inv)
{
  for (int sector = 0; sector < o2::tpc::constants::MAXSECTOR; sector++) {
    for (int p = 0; p < TPC_PADS_IN_SECTOR; p++) {
      const float gainVal = gainMap.getValue(sector, p);
      inv ? mGainCorrection[sector].set(p, (gainVal > 1.e-5f) ? 1.f / gainVal : 1.f) : mGainCorrection[sector].set(p, gainVal);
    }
  }
}
#endif

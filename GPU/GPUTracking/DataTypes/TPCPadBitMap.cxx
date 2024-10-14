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

/// \file TPCPadBitMap.cxx
/// \author Jens Wiechula

#include "TPCPadBitMap.h"

#include "GPUTPCGeometry.h"
#include "DataFormatsTPC/Constants.h"

using namespace GPUCA_NAMESPACE::gpu;

TPCPadBitMap::TPCPadBitMap()
{
  GPUTPCGeometry geo{};
  int32_t offset = 0;
  for (int32_t r = 0; r < GPUCA_ROW_COUNT; r++) {
    mPadOffsetPerRow[r] = offset;
    offset += geo.NPads(r);
  }
}

#ifndef GPUCA_STANDALONE
#include "TPCBase/CalDet.h"

TPCPadBitMap::TPCPadBitMap(const o2::tpc::CalDet<bool>& map) : TPCPadBitMap()
{
  setFromMap(map);
}

void TPCPadBitMap::setFromMap(const o2::tpc::CalDet<bool>& map)
{
  for (int32_t sector = 0; sector < o2::tpc::constants::MAXSECTOR; sector++) {
    for (int32_t p = 0; p < TPC_PADS_IN_SECTOR; p++) {
      const auto val = map.getValue(sector, p);
      mBitMap[sector].set(p, val);
    }
  }
}
#endif

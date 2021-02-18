// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TPCPadGainCalib.cxx
/// \author Felix Weiglhofer

#include "TPCPadGainCalib.h"

#include "GPUTPCGeometry.h"
#include "DataFormatsTPC/Constants.h"
#include "TPCBase/CalDet.h"

using namespace GPUCA_NAMESPACE::gpu;

TPCPadGainCalib::TPCPadGainCalib()
{
  GPUTPCGeometry geo{};
  int offset = 0;
  for (int r = 0; r < TPC_NUM_OF_ROWS; r++) {
    mPadOffsetPerRow[r] = offset;
    offset += geo.NPads(r);
  }
}

TPCPadGainCalib::TPCPadGainCalib(const o2::tpc::CalDet<float>& gainMap) : TPCPadGainCalib()
{
  for (int sector = 0; sector < o2::tpc::constants::MAXSECTOR; sector++) {
    for (int p = 0; p < TPC_PADS_IN_SECTOR; p++) {
      mGainCorrection[sector].set(p, gainMap.getValue(sector, p));
    }
  }
}

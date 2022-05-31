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

/// \file StatusHelper.cxx
/// \brief Helper with 'permanent' status of TRD half-chambers to support QC plots
/// \author Ole Schmidt

#include "TRDQC/StatusHelper.h"
#include "DataFormatsTRD/HelperMethods.h"
#include <fairlogger/Logger.h>

using namespace o2::trd;

void HalfChamberStatusQC::maskChamber(int sec, int stack, int ly)
{
  int det = HelperMethods::getDetector(ly, stack, sec);
  mStatus.set(det * 2);
  mStatus.set(det * 2 + 1);
}

void HalfChamberStatusQC::maskHalfChamberA(int sec, int stack, int ly)
{
  int det = HelperMethods::getDetector(ly, stack, sec);
  mStatus.set(det * 2);
}

void HalfChamberStatusQC::maskHalfChamberB(int sec, int stack, int ly)
{
  int det = HelperMethods::getDetector(ly, stack, sec);
  mStatus.set(det * 2 + 1);
}

void HalfChamberStatusQC::print()
{
  LOG(info) << "The following half-chambers are masked:";
  for (int iHc = 0; iHc < constants::MAXHALFCHAMBER; ++iHc) {
    if (mStatus.test(iHc)) {
      char side = (iHc % 2 == 0) ? 'A' : 'B'; // 0 - side A, 1 - side B
      int det = iHc / 2;
      int sec = HelperMethods::getSector(det);
      int stack = HelperMethods::getStack(det);
      int layer = HelperMethods::getLayer(det);
      if (side == 'A' && mStatus.test(iHc + 1)) {
        // full chamber is masked
        LOGF(info, "%02d_%d_%d", sec, stack, layer);
        iHc++;
      } else {
        LOGF(info, "%02d_%d_%d_%c", sec, stack, layer, side);
      }
    }
  }
}

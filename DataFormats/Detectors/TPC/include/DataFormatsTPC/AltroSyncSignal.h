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

/// \file AltroSyncSignal.h
/// \brief Definition of the timebin from which syncronization starts

#ifndef ALICEO2_DATAFORMATSTPC_ALTROSYNCSIGNAL_H
#define ALICEO2_DATAFORMATSTPC_ALTROSYNCSIGNAL_H

#include "GPUCommonRtypes.h"

namespace o2::tpc
{
struct AltroSyncSignal {
  int periodTF = 10;     // signal repeats every period-th TF
  int timebin = 14192;   // every 10 TF, orbit 31, Time bin 384, BC 4 -> 14195, but clusters can be affected before that

  int getTB2Cut(uint32_t tfCounter) const
  {
    return periodTF > 0 && (tfCounter % periodTF) == 1 && tfCounter > periodTF ? timebin : -1;
  }

  ClassDefNV(AltroSyncSignal, 1);
};
} // namespace o2::tpc

#endif // ALICEO2_DATAFORMATSTPC_ALTROSYNCSIGNAL_H

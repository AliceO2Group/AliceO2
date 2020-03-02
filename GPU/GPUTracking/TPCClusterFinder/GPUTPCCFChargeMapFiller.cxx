// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChargeMapFiller.cxx
/// \author Felix Weiglhofer

#include "GPUTPCCFChargeMapFiller.h"
#include "ChargePos.h"
#include "Array2D.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::deprecated;

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::fillChargeMap>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  fillChargeMapImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPdigits, chargeMap, clusterer.mPmemory->counters.nDigits);
}

GPUd() void GPUTPCCFChargeMapFiller::fillChargeMapImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                       const Digit* digits,
                                                       Array2D<PackedCharge>& chargeMap,
                                                       size_t maxDigit)
{
  size_t idx = get_global_id(0);
  if (idx >= maxDigit) {
    return;
  }
  Digit myDigit = digits[idx];

  chargeMap[ChargePos(myDigit)] = PackedCharge(myDigit.charge);
}

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::resetMaps>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  resetMapsImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPdigits, chargeMap, isPeakMap);
}

GPUd() void GPUTPCCFChargeMapFiller::resetMapsImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                   const Digit* digits,
                                                   Array2D<PackedCharge>& chargeMap,
                                                   Array2D<uchar>& isPeakMap)
{
  size_t idx = get_global_id(0);
  Digit myDigit = digits[idx];

  ChargePos pos(myDigit);

  chargeMap[pos] = PackedCharge(0);
  isPeakMap[pos] = 0;
}

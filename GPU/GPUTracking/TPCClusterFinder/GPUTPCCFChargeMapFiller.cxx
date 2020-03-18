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

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::deprecated;

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::fillIndexMap>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<uint> indexMap(clusterer.mPindexMap);
  fillIndexMapImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPpositions, indexMap, clusterer.mPmemory->counters.nDigits);
}

GPUd() void GPUTPCCFChargeMapFiller::fillIndexMapImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                      const ChargePos* positions,
                                                      Array2D<uint>& indexMap,
                                                      size_t maxDigit)
{
  size_t idx = get_global_id(0);
  if (idx >= maxDigit) {
    return;
  }
  CPU_ONLY(ChargePos pos = positions[idx]);
  CPU_ONLY(indexMap.safeWrite(pos, idx));
}

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::fillFromDigits>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  fillFromDigitsImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPdigits, clusterer.mPpositions, chargeMap, clusterer.mPmemory->counters.nDigits);
}

GPUd() void GPUTPCCFChargeMapFiller::fillFromDigitsImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                        const Digit* digits,
                                                        ChargePos* positions,
                                                        Array2D<PackedCharge>& chargeMap,
                                                        size_t maxDigit)
{
  size_t idx = get_global_id(0);
  if (idx >= maxDigit) {
    return;
  }
  Digit digit = digits[idx];
  ChargePos pos(digit.row, digit.pad, digit.time);
  positions[idx] = pos;
  chargeMap[pos] = PackedCharge(digit.charge);
}

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::resetMaps>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  Array2D<uchar> isPeakMap(clusterer.mPpeakMap);
  resetMapsImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPpositions, chargeMap, isPeakMap, clusterer.mPmemory->counters.nDigits);
}

GPUd() void GPUTPCCFChargeMapFiller::resetMapsImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                   const ChargePos* positions,
                                                   Array2D<PackedCharge>& chargeMap,
                                                   Array2D<uchar>& isPeakMap,
                                                   size_t maxDigit)
{
  size_t idx = get_global_id(0);
  if (idx >= maxDigit) {
    return;
  }

  ChargePos pos = positions[idx];

  chargeMap[pos] = PackedCharge(0);
  isPeakMap[pos] = 0;
}

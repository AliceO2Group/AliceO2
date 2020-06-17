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
#include "DataFormatsTPC/Digit.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::fillIndexMap>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<uint> indexMap(clusterer.mPindexMap);
  fillIndexMapImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPmemory->fragment, clusterer.mPdigits, indexMap, clusterer.mPmemory->counters.nPositions);
}

GPUd() void GPUTPCCFChargeMapFiller::fillIndexMapImpl(int nBlocks, int nThreads, int iBlock, int iThread,
                                                      const CfFragment& fragment,
                                                      const tpc::Digit* digits,
                                                      Array2D<uint>& indexMap,
                                                      size_t maxDigit)
{
  size_t idx = get_global_id(0);
  if (idx >= maxDigit) {
    return;
  }
  CPU_ONLY(idx += fragment.digitsStart);
  CPU_ONLY(tpc::Digit digit = digits[idx]);
  CPU_ONLY(ChargePos pos(digit.getRow(), digit.getPad(), fragment.toLocal(digit.getTimeStamp())));
  CPU_ONLY(indexMap.safeWrite(pos, idx));
}

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::fillFromDigits>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  Array2D<PackedCharge> chargeMap(reinterpret_cast<PackedCharge*>(clusterer.mPchargeMap));
  fillFromDigitsImpl(get_num_groups(0), get_local_size(0), get_group_id(0), get_local_id(0), clusterer.mPmemory->fragment, clusterer.mPmemory->counters.nPositions, clusterer.mPdigits, clusterer.mPpositions, chargeMap);
}

GPUd() void GPUTPCCFChargeMapFiller::fillFromDigitsImpl(int nBlocks, int nThreads, int iBlock, int iThread, const CfFragment& fragment, size_t digitNum,
                                                        const tpc::Digit* digits,
                                                        ChargePos* positions,
                                                        Array2D<PackedCharge>& chargeMap)
{
  size_t idx = get_global_id(0);
  if (idx >= digitNum) {
    return;
  }
  tpc::Digit digit = digits[fragment.digitsStart + idx];

  ChargePos pos(digit.getRow(), digit.getPad(), fragment.toLocal(digit.getTimeStamp()));
  positions[idx] = pos;
  chargeMap[pos] = PackedCharge(digit.getChargeFloat());
}

template <>
GPUdii() void GPUTPCCFChargeMapFiller::Thread<GPUTPCCFChargeMapFiller::findFragmentStart>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer)
{
  // TODO: use binary search
  if (get_global_id(0) != 0) {
    return;
  }

  size_t nDigits = clusterer.mPmemory->counters.nDigits;
  const tpc::Digit* digits = clusterer.mPdigits;
  size_t st = 0;
  for (; st < nDigits && digits[st].getTimeStamp() < clusterer.mPmemory->fragment.first(); st++) {
  }

  size_t end = st;
  for (; end < nDigits && digits[end].getTimeStamp() < clusterer.mPmemory->fragment.last(); end++) {
  }

  clusterer.mPmemory->fragment.digitsStart = st;
  clusterer.mPmemory->counters.nPositions = end - st;
}
